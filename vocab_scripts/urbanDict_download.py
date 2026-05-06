import asyncio
import aiohttp
import requests
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('urban_dict_download.log'),
        logging.StreamHandler()
    ]
)

class UrbanDictDownloader:
    def __init__(self, output_dir, cutoff_date="2015-01-01", max_concurrent=20, delay_between_batches=1.0):
        self.api_base = "http://api.urbandictionary.com/v0"
        self.cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
        self.max_concurrent = max_concurrent
        self.delay_between_batches = delay_between_batches
        self.vocab_pre2015 = set()
        self.failed_words = set()
        self.processed_words = set()
        self.lock = threading.Lock()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoint_urban_dict.json"
        self.output_file = self.output_dir / "urban_dict_pre2015_vocab.txt"
        self.failed_words_file = self.output_dir / "urban_dict_failed_words.txt"

    def download_word_list(self):
        logging.info("Downloading word list from GitHub...")

        base_url = "https://raw.githubusercontent.com/mattbierner/urban-dictionary-word-list/master/data/"
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        all_words = []
        for letter in letters:
            url = f"{base_url}{letter}.data"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                words = response.text.strip().split('\n')
                all_words.extend(words)
                logging.info(f"Downloaded {len(words)} words starting with '{letter}'")
                time.sleep(0.3)
            except Exception as e:
                logging.error(f"Error downloading words for letter {letter}: {e}")

        logging.info(f"Total words downloaded: {len(all_words)}")
        return all_words

    def load_checkpoint(self):
        if Path(self.checkpoint_file).exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.vocab_pre2015 = set(data.get('vocab', []))
                self.failed_words = set(data.get('failed', []))
                self.processed_words = set(data.get('processed', []))
                logging.info(f"Loaded checkpoint: {len(self.vocab_pre2015)} vocab, {len(self.failed_words)} failed, {len(self.processed_words)} processed")

    def save_checkpoint(self):
        with self.lock:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'vocab': list(self.vocab_pre2015),
                    'failed': list(self.failed_words),
                    'processed': list(self.processed_words)
                }, f)

    def filter_pre2015(self, word, api_response):
        if not api_response or 'list' not in api_response:
            return False

        for definition in api_response['list']:
            written_on = definition.get('written_on')
            if written_on:
                try:
                    date_str = written_on.replace('Z', '').replace('T', ' ')
                    date_str = date_str.split('.')[0]
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    if date < self.cutoff_date:
                        return True
                except Exception:
                    pass

        return False

    async def query_word_async(self, session, word):
        url = f"{self.api_base}/define"
        params = {'term': word}

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return word, data, False
                else:
                    return word, None, response.status >= 500
        except asyncio.TimeoutError:
            return word, None, True
        except Exception as e:
            is_server_error = "500" in str(e) or "503" in str(e) or "502" in str(e)
            return word, None, is_server_error

    async def process_batch(self, words):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.query_word_async(session, word) for word in words]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue

                word, api_response, is_server_error = result

                with self.lock:
                    if api_response and self.filter_pre2015(word, api_response):
                        self.vocab_pre2015.add(word)
                    elif is_server_error:
                        self.failed_words.add(word)

                    self.processed_words.add(word)

    def download_and_filter(self):
        all_words = self.download_word_list()

        self.load_checkpoint()

        words_to_process = [w for w in all_words if w not in self.vocab_pre2015]
        total_words = len(words_to_process)

        logging.info(f"Starting processing of {total_words} remaining words with {self.max_concurrent} concurrent requests...")

        batch_size = self.max_concurrent * 5
        batches = [words_to_process[i:i+batch_size] for i in range(0, len(words_to_process), batch_size)]

        start_time = time.time()
        processed_count = 0

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for batch_idx, batch in enumerate(batches):
            loop.run_until_complete(self.process_batch(batch))

            processed_count += len(batch)

            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_words - processed_count) / rate if rate > 0 else 0

            logging.info(
                f"Batch {batch_idx+1}/{len(batches)} - "
                f"Processed: {processed_count}/{total_words} ({processed_count/total_words*100:.1f}%) - "
                f"Found: {len(self.vocab_pre2015)} pre-2015 - "
                f"Rate: {rate:.1f} words/sec - "
                f"ETA: {eta/60:.1f} min"
            )

            if batch_idx % 10 == 0:
                self.save_checkpoint()

            time.sleep(self.delay_between_batches)

        loop.close()

        self.save_checkpoint()
        self.save_vocabulary()
        self.save_failed_words()

        total_time = time.time() - start_time
        logging.info(f"Complete! Found {len(self.vocab_pre2015)} pre-2015 words in {total_time/60:.1f} minutes")

    def save_vocabulary(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for word in sorted(self.vocab_pre2015):
                f.write(f"{word}\n")
        logging.info(f"Vocabulary saved to {self.output_file}")

    def save_failed_words(self):
        with open(self.failed_words_file, 'w', encoding='utf-8') as f:
            for word in sorted(self.failed_words):
                f.write(f"{word}\n")
        logging.info(f"Failed words saved to {self.failed_words_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Urban Dictionary Pre-2015 Vocab Extractor")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output vocab files")
    parser.add_argument("--concurrent", type=int, default=20, help="Number of concurrent requests (default: 20)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between batches in seconds (default: 1.0)")
    args = parser.parse_args()

    downloader = UrbanDictDownloader(
        output_dir=args.output_dir,
        cutoff_date="2015-01-01",
        max_concurrent=args.concurrent,
        delay_between_batches=args.delay
    )

    downloader.download_and_filter()
