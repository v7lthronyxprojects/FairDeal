import os
import re
import logging
import requests
import socket
import socks
from time import sleep
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QProgressBar,
    QStatusBar, QCheckBox, QGroupBox, QFrame, QGridLayout,
    QScrollArea
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import cv2
import pytesseract
from stem import Signal
from stem.control import Controller
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

SITES = [
    {"name": "Digikala", "url": "digikala.com", "category": "نو"},
    {"name": "Banimode", "url": "banimode.com", "category": "نو"},
    {"name": "Modiseh", "url": "modiseh.com", "category": "نو"},
    {"name": "Khanoumi", "url": "khanoumi.com", "category": "نو"},
    {"name": "Meghdad IT", "url": "meghdadit.com", "category": "نو"},
    {"name": "Lion Computer", "url": "lioncomputer.com", "category": "نو"},
    {"name": "Torob", "url": "torob.com", "category": "نو"},
    {"name": "Emalls", "url": "emalls.ir", "category": "نو"},
    {"name": "Shixon", "url": "shixon.com", "category": "نو"},
    {"name": "DigiStyle", "url": "digistyle.com", "category": "نو"},
    {"name": "Snapp Market", "url": "snapp.market", "category": "نو"},
    {"name": "دیوار", "url": "divar.ir", "category": "دست دوم"},
    {"name": "شیپور", "url": "sheypoor.com", "category": "دست دوم"},
    {"name": "ایسام", "url": "esam.ir", "category": "دست دوم"},
    {"name": "کمدا", "url": "komodaa.com", "category": "دست دوم"},
    {"name": "پیندو", "url": "pindo.ir", "category": "دست دوم"},
    {"name": "ریباکس", "url": "rebox.ir", "category": "دست دوم"},
    {"name": "نوبازار", "url": "nobazaar.ir", "category": "دست دوم"},
    {"name": "تخفیفان", "url": "takhfifan.com", "category": "دست دوم"},
    {"name": "باما", "url": "bama.ir", "category": "دست دوم"},
    {"name": "چارسو", "url": "charsooq.com", "category": "دست دوم"},
    {"name": "آی‌تی بازار", "url": "itbazar.com", "category": "نو"},
    {"name": "دیجی‌کالا یوزد", "url": "used.digikala.com", "category": "دست دوم"}
]

@lru_cache(maxsize=100)
def cached_search(query: str, site_url: str) -> list:
    return []

def extract_price(snippet: str) -> float:
    try:
        if isinstance(snippet, (int, float)):
            return float(snippet)

        snippet = snippet.replace('٬', ',').replace('،', ',')
        persian_nums = '۰۱۲۳۴۵۶۷۸۹'
        arabic_nums = '٠١٢٣٤٥٦٧٨٩'
        trans = str.maketrans(persian_nums + arabic_nums, '0123456789' * 2)
        snippet = snippet.translate(trans)

        currency_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\s*(?:هزار|میلیون|میلیارد))?)\s*(?:تومان|ریال|تومن)',
            r'(\d{1,3}(?:,\d{3})*)\s*ت',
            r'(\d{1,3}(?:,\d{3})*)\s*ر(?:یال)?'
        ]

        prices = []
        for pattern in currency_patterns:
            matches = re.finditer(pattern, snippet, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    price_str = match.group(1).replace(',', '')
                    price = float(price_str)

                    if 'هزار' in match.group():
                        price *= 1_000
                    elif 'میلیون' in match.group():
                        price *= 1_000_000
                    elif 'میلیارد' in match.group():
                        price *= 1_000_000_000

                    if 'ریال' in match.group() or 'ر' in match.group():
                        price /= 10

                    if 100 <= price <= 1_000_000_000_000:
                        prices.append(price)
                except ValueError:
                    continue

        if prices:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            valid_prices = [p for p in prices if abs(p - mean_price) <= 2 * std_price]

            if valid_prices:
                return min(valid_prices)

        return float('inf')

    except Exception as e:
        logging.error(f"Price extraction error: {e}")
        return float('inf')

def filter_results(results: list, query: str) -> list:
    filtered_results = []
    query_terms = set(query.lower().strip().split())
    required_terms = {term for term in query_terms if len(term) > 2}

    for result in results:
        if not isinstance(result, dict):
            continue

        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()

        title_terms = set(re.sub(r'[^\w\s]', ' ', title).split())

        matching_terms = required_terms.intersection(title_terms)
        match_score = len(matching_terms) / len(required_terms) if required_terms else 0

        exact_match = query.lower() in title.lower()
        all_terms_present = len(matching_terms) == len(required_terms)
        has_valid_price = isinstance(result.get('price'), (int, float)) and result['price'] != float('inf')

        if (exact_match or (match_score >= 0.8 and all_terms_present)) and has_valid_price:
            result['match_score'] = match_score
            result['exact_match'] = exact_match
            filtered_results.append(result)

    return sorted(
        filtered_results,
        key=lambda x: (
            x.get('exact_match', False),
            x.get('match_score', 0),
            float(x.get('price', float('inf')))
        ),
        reverse=True
    )

def is_relevant_product(title: str, query: str) -> bool:
    query = query.lower().strip()
    title = title.lower().strip()

    common_words = {'خرید', 'قیمت', 'فروش', 'انواع', 'مدل', 'جدید', 'اصل', 'اورجینال'}
    query_terms = {term for term in query.split() if term not in common_words and len(term) > 2}

    if query in title:
        return True

    title_terms = set(re.sub(r'[^\w\s]', ' ', title).split())
    matching_terms = query_terms.intersection(title_terms)

    return len(matching_terms) == len(query_terms)

def check_availability(snippet: str) -> bool:
    available_patterns = [
        'موجود',
        'در انبار',
        'قابل خرید',
        'در دسترس',
        'امکان خرید',
        'قابل سفارش',
        'افزودن به سبد',
        'خرید محصول',
        'سفارش محصول',
        'اضافه به سبد',
    ]
    unavailable_patterns = [
        'ناموجود',
        'اتمام',
        'تمام شد',
        'موجود نیست',
        'به زودی',
        'پایان موجودی',
        'توقف تولید',
        'در دسترس نیست',
        'فعلا موجود نیست',
        'تمام شده'
    ]

    snippet = snippet.lower()

    for pattern in unavailable_patterns:
        if pattern in snippet:
            return False

    for pattern in available_patterns:
        if pattern in snippet:
            return True

    price_text = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\s*تومان)?)', snippet)
    if price_text:
        return True

    return False

def sorted_results(results: list) -> dict:
    try:
        if not results:
            return {"نو": [], "دست دوم": []}

        for result in results:
            if isinstance(result, dict):
                if 'price' not in result or not result['price']:
                    result['price'] = float('inf')
                elif isinstance(result['price'], str):
                    result['price'] = extract_price(result['price'])

        categorized = {"نو": [], "دست دوم": []}
        for result in results:
            if isinstance(result, dict):
                category = result.get('category', "دست دوم")
                categorized.setdefault(category, []).append(result)

        for key in categorized:
            categorized[key] = sorted(categorized[key], key=lambda x: float(x.get('price', float('inf'))))

        return categorized
    except Exception as e:
        logging.error(f"Error in sorted_results: {e}")
        return {"نو": [], "دست دوم": []}

class AdaptiveMLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.load_or_create_models()

    def load_or_create_models(self):
        try:
            self.vectorizer = load('vectorizer.joblib')
            self.price_model = load('price_model.joblib')
            self.category_model = load('category_model.joblib')
            self.scaler = load('scaler.joblib')
            self.is_fitted = True
            logging.info("Models loaded successfully.")
        except Exception as e:
            logging.warning(f"Unable to load models, creating new ones: {e}")
            self.train_initial_models()

    def train_initial_models(self):
        try:
            default_data = pd.DataFrame({
                'title': [
                    'گوشی موبایل سامسونگ',
                    'لپ تاپ ایسوس',
                    'تبلت اپل',
                    'ساعت هوشمند شیائومی',
                    'هدفون بلوتوث سونی'
                ],
                'snippet': [
                    'گوشی هوشمند جدید با قابلیت 5G',
                    'لپ تاپ گیمینگ با پردازنده قوی',
                    'تبلت با صفحه نمایش رتینا',
                    'ساعت هوشمند با عمر باتری طولانی',
                    'هدفون با کیفیت صدای عالی'
                ],
                'price': [
                    5_000_000,
                    15_000_000,
                    12_000_000,
                    2_000_000,
                    1_500_000
                ],
                'category': ['نو', 'نو', 'نو', 'نو', 'دست دوم']
            })

            X = self.vectorizer.fit_transform(default_data['title'] + ' ' + default_data['snippet'])
            X_dense = X.toarray()

            X_scaled = self.scaler.fit_transform(X_dense)

            self.price_model.fit(X_scaled, default_data['price'])
            self.category_model.fit(X_scaled, default_data['category'])

            self.is_fitted = True
            self.save_models()
            logging.info("Initial models trained successfully with default data.")

        except Exception as e:
            logging.error(f"Error in training initial models: {e}")
            self.is_fitted = False

    def predict(self, title: str, snippet: str):
        try:
            if not self.is_fitted:
                return float('inf'), "نو"

            X = self.vectorizer.transform([title + ' ' + snippet])
            X_dense = X.toarray()

            X_scaled = self.scaler.transform(X_dense)

            price = self.price_model.predict(X_scaled)[0]
            category = self.category_model.predict(X_scaled)[0]

            return price, category

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return float('inf'), "نو"

    def update_models(self, new_data: pd.DataFrame):
        if new_data.empty:
            return

        try:
            X = self.vectorizer.transform(new_data['title'] + ' ' + new_data['snippet'])
            X_dense = X.toarray()

            X_scaled = self.scaler.transform(X_dense)

            if not self.is_fitted:
                self.price_model.fit(X_scaled, new_data['price'])
                self.category_model.fit(X_scaled, new_data['category'])
                self.is_fitted = True
            else:
                self.price_model.fit(X_scaled, new_data['price'])
                self.category_model.fit(X_scaled, new_data['category'])

            self.save_models()
            logging.info("Machine learning models updated successfully with new data.")

        except Exception as e:
            logging.error(f"Error updating models: {e}")

    def save_models(self):
        try:
            dump(self.vectorizer, 'vectorizer.joblib')
            dump(self.price_model, 'price_model.joblib')
            dump(self.category_model, 'category_model.joblib')
            dump(self.scaler, 'scaler.joblib')
            logging.info("Models saved successfully.")
        except Exception as e:
            logging.error(f"Error saving models: {e}")

class Worker(QThread):
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    status_message = pyqtSignal(str)

    def __init__(self, product_name: str, use_tor: bool = False, search_methods: list = None):
        super().__init__()
        self.product_name = product_name
        self._is_running = True
        self.ml_model = AdaptiveMLModel()
        self.collected_data = []
        self.use_tor = use_tor
        self.search_methods = search_methods or ["api", "scrape", "selenium"]

        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        if self.use_tor:
            self.enable_tor()

        self.selenium_options = {
            'page_load_timeout': 20,
            'implicit_wait': 10,
            'headless': True,
            'disable_gpu': True,
            'no_sandbox': True,
            'disable_dev_shm': True
        }
        
        self.request_timeout = 15
        self.max_retries = 3
        self.retry_delay = 2

    def enable_tor(self):
        try:
            with Controller.from_port(port=9051) as controller:
                controller.authenticate()
                if not controller.is_newnym_available():
                    raise Exception("Tor is not ready for new circuits")

            socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 9050)
            socket.socket = socks.socksocket

            test_sock = socks.socksocket()
            test_sock.connect(("www.google.com", 80))
            test_sock.close()

            self.new_tor_identity()

            self.status_message.emit("Tor با موفقیت فعال شد")
            logging.info("Tor successfully enabled.")
        except Exception as e:
            logging.error(f"Error enabling Tor: {e}")
            self.status_message.emit("خطا در راه‌اندازی Tor - لطفاً از فعال بودن سرویس Tor اطمینان حاصل کنید")
            self.use_tor = False

    def new_tor_identity(self):
        try:
            with Controller.from_port(port=9051) as controller:
                try:
                    controller.authenticate()
                except Exception:
                    try:
                        controller.authenticate(password="")
                    except Exception:
                        controller.authenticate(password="password")

                if controller.is_newnym_available():
                    controller.signal(Signal.NEWNYM)
                    sleep(controller.get_newnym_wait())
                    self.status_message.emit("هویت جدید Tor دریافت شد")
                    logging.info("New Tor identity acquired.")
                else:
                    self.status_message.emit("در حال حاضر امکان تغییر هویت Tor وجود ندارد")
                    logging.warning("Cannot change Tor identity at the moment.")
        except Exception as e:
            logging.error(f"Error getting new Tor identity: {e}")
            self.status_message.emit("خطا در تغییر هویت Tor")
            self.use_tor = False

    def stop(self):
        self._is_running = False
        self.terminate()
        self.status_message.emit("جستجو متوقف شد")
        logging.info("Search stopped.")

    def run(self):
        try:
            if not self.ml_model:
                self.result_ready.emit({"نو": [], "دست دوم": []})
                return

            results = []
            total_sites = len(SITES)
            completed = 0

            if "selenium" in self.search_methods:
                self.status_message.emit("در حال جستجوی مستقیم در سایت‌ها...")
                for site in SITES:
                    if not self._is_running:
                        self.result_ready.emit({"نو": [], "دست دوم": []})
                        return

                    try:
                        direct_results = self.direct_site_search(self.product_name, site)
                        if direct_results:
                            results.extend(direct_results)
                    except Exception as e:
                        logging.error(f"Error in direct search for {site['name']}: {e}")

                    completed += 1
                    progress = int((completed / total_sites) * 50)
                    self.progress.emit(progress)

            if len(results) < 5 or "selenium" not in self.search_methods:
                methods = self.get_search_methods()
                method_names = {
                    self.google_api_search: "Google API",
                    self.google_scrape_search: "Web Scraping",
                    self.selenium_scrape_search: "Selenium"
                }

                self.status_message.emit(f"جستجو با روش‌های: {', '.join([method_names[m] for m in methods])}")

                for site in SITES:
                    if not self._is_running:
                        break

                    for method in methods:
                        try:
                            method_results = method(self.product_name, site)
                            if method_results:
                                results.extend(method_results)
                        except Exception as e:
                            logging.error(f"Error with {method.__name__} for {site['name']}: {e}")

                    completed += 1
                    progress = 50 + int((completed / total_sites) * 50)
                    self.progress.emit(progress)

            results = filter_results(results, self.product_name)
            unique_results = self.remove_duplicates(results)
            final_results = sorted_results(unique_results)

            self.update_ml_models()

            self.result_ready.emit(final_results)
            self.status_message.emit("جستجو به پایان رسید.")

        except Exception as e:
            logging.error(f"Error in search: {e}")
            self.result_ready.emit({"نو": [], "دست دوم": []})

    def direct_site_search(self, query: str, site: dict) -> list:
        results = []
        try:
            base_url = f"https://{site['url']}"

            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-software-rasterizer')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--log-level=3')
            options.add_argument('--silent')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])

            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            driver.set_page_load_timeout(30)

            try:
                search_urls = {
                    'digikala.com': f"{base_url}/search/?q={query}",
                    'divar.ir': f"{base_url}/s/tehran?q={query}",
                    'torob.com': f"{base_url}/search/?query={query}",
                }

                search_url = search_urls.get(site['url'], f"{base_url}/search?q={query}")
                driver.get(search_url)

                sleep(2)

                products = driver.find_elements(By.CSS_SELECTOR, '[class*="product"], [class*="card"], [class*="item"]')

                for product in products[:10]:
                    try:
                        title = product.find_element(By.CSS_SELECTOR, '[class*="title"], h2, h3').text.strip()

                        if not is_relevant_product(title, query):
                            continue

                        price_elem = product.find_element(By.CSS_SELECTOR, '[class*="price"]')
                        price = extract_price(price_elem.text)

                        if price == float('inf'):
                            continue

                        link = product.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')

                        try:
                            description = product.find_element(By.CSS_SELECTOR, 
                                '[class*="description"], [class*="specs"], [class*="details"]').text.strip()
                        except:
                            description = ''

                        result = {
                            'title': title,
                            'link': link,
                            'snippet': description,
                            'price': price,
                            'category': site['category'],
                            'site': site,
                            'source': 'Direct',
                            'availability': True
                        }

                        results.append(result)

                    except Exception as e:
                        logging.error(f"Error extracting product info: {e}")
                        continue

            except Exception as e:
                logging.error(f"Error accessing {site['url']}: {e}")

            finally:
                driver.quit()

        except Exception as e:
            logging.error(f"Error in direct site search for {site['url']}: {e}")

        return results

    def get_search_methods(self):
        methods = []
        if "api" in self.search_methods:
            methods.append(self.google_api_search)
        if "scrape" in self.search_methods:
            methods.append(self.google_scrape_search)
        if "selenium" in self.search_methods:
            methods.append(self.selenium_scrape_search)
        return methods

    def remove_duplicates(self, results: list) -> list:
        seen_urls = set()
        unique_results = []

        for result in results:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        return unique_results

    def retry_search(self) -> list:
        results = []
        search_queries = [
            self.product_name,
            f"قیمت {self.product_name}",
            f"خرید {self.product_name}",
            f"{self.product_name} فروش"
        ]

        for query in search_queries:
            for site in SITES:
                try:
                    scrape_results = self.google_scrape_search(query, site)
                    if scrape_results:
                        results.extend(scrape_results)
                except Exception as e:
                    logging.error(f"Retry search error for {site['name']}: {e}")
                    continue

        return results

    def google_api_search(self, query: str, site: dict) -> list:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": f"site:{site['url']} {query}",
            "num": 10
        }
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self.parse_api_results(response.json(), site)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.warning(f"Rate limit exceeded for {site['name']}. Switching to scraping...")
                return self.google_scrape_search(query, site)
            logging.error(f"API search error for {site['name']}: {e}")
            return []
        except Exception as e:
            logging.error(f"API search exception for {site['name']}: {e}")
            return []

    def google_scrape_search(self, query: str, site: dict) -> list:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        search_url = f"https://www.google.com/search?q=site:{site['url']}+{query}&num=20"
        try:
            response = self.session.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            return self.parse_scrape_results(BeautifulSoup(response.text, 'html.parser'), site)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.warning(f"Rate limit exceeded for {site['name']}. Switching to Selenium...")
                return self.selenium_scrape_search(query, site)
            logging.error(f"Scrape search error for {site['name']}: {e}")
            return []
        except Exception as e:
            logging.error(f"Scrape search exception for {site['name']}: {e}")
            return []

    def selenium_scrape_search(self, query: str, site: dict) -> list:
        options = Options()
        for key, value in self.selenium_options.items():
            if isinstance(value, bool):
                if value:
                    options.add_argument(f'--{key}')
            else:
                options.set_capability(key, value)

        try:
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            try:
                driver.set_page_load_timeout(self.selenium_options['page_load_timeout'])
                driver.implicitly_wait(self.selenium_options['implicit_wait'])

                return self.parallel_search_execution(driver, query, site)
            except Exception as e:
                logging.error(f"Selenium error: {e}")
                return []
            finally:
                driver.quit()
        except Exception as e:
            logging.error(f"Error initializing Selenium WebDriver: {e}")
            return []

    def parallel_search_execution(self, driver, query: str, site: dict) -> list:
        from concurrent.futures import ThreadPoolExecutor

        search_functions = [
            self.search_main_page,
            self.search_category_pages,
            self.search_product_pages
        ]

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(func, driver, query, site) 
                for func in search_functions
            ]

            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.extend(result)
                except Exception as e:
                    logging.error(f"Parallel search error: {e}")

        return results

    def parse_api_results(self, json_data: dict, site: dict) -> list:
        results = []
        try:
            items = json_data.get('items', [])
            for item in items:
                title = item.get('title', 'بدون عنوان').strip()
                link = item.get('link', '#').strip()
                snippet = item.get('snippet', '').strip()

                if not is_relevant_product(title, self.product_name) or link == '#' or not title:
                    continue

                price = extract_price(snippet)
                availability = check_availability(snippet)

                predicted_price, predicted_category = self.ml_model.predict(title, snippet)
                final_price = min(price, predicted_price) if predicted_price != float('inf') else price

                category = next((s['category'] for s in SITES if s['url'] in link), predicted_category)

                result = {
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                    'price': final_price,
                    'category': category,
                    'site': site,
                    'source': 'API',
                    'availability': availability if availability is not None else True
                }

                self.collect_training_data(title, snippet, final_price, category)
                results.append(result)

        except Exception as e:
            logging.error(f"Error parsing API results: {e}")

        return results

    def parse_scrape_results(self, soup: BeautifulSoup, site: dict) -> list:
        search_results = []
        try:
            for g in soup.find_all('div', class_='tF2Cxc'):
                title_tag = g.find('h3')
                title = title_tag.text.strip() if title_tag else 'نامشخص'
                link = g.find('a')['href'].strip() if g.find('a') else '#'
                snippet_tag = g.find('span', class_='aCOpRe')
                snippet = snippet_tag.text.strip() if snippet_tag else ''

                if not is_relevant_product(title, self.product_name) or link == '#' or not title:
                    continue

                availability = check_availability(snippet)
                initial_price = extract_price(snippet)

                if availability is not False and initial_price != float('inf'):
                    predicted_price, predicted_category = self.ml_model.predict(title, snippet)
                    price = min(initial_price, predicted_price) if predicted_price != float('inf') else initial_price

                    category = next((s['category'] for s in SITES if s['url'] in link), predicted_category)

                    result = {
                        'title': title,
                        'link': link,
                        'snippet': snippet,
                        'price': price,
                        'category': category,
                        'site': site,
                        'source': 'Scraping',
                        'availability': availability if availability is not None else True
                    }

                    if price != float('inf'):
                        self.collect_training_data(title, snippet, price, category)
                        search_results.append(result)

        except Exception as e:
            logging.error(f"Error parsing search result: {e}")

        return search_results

    def collect_training_data(self, title: str, snippet: str, price: float, category: str):
        self.collected_data.append({
            'title': title,
            'snippet': snippet,
            'price': price,
            'category': category
        })

    def update_ml_models(self):
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            self.ml_model.update_models(df)
            self.collected_data = []
            logging.info("Machine learning models updated with new data.")

    def solve_captcha(self, image_path: str) -> str:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            image = cv2.medianBlur(image, 3)

            captcha_text = pytesseract.image_to_string(image, config='--psm 8')
            return captcha_text.strip()
        except Exception as e:
            logging.error(f"Error solving captcha: {e}")
            return ""

    def search_main_page(self, driver, query: str, site: dict) -> list:
        return []

    def search_category_pages(self, driver, query: str, site: dict) -> list:
        return []

    def search_product_pages(self, driver, query: str, site: dict) -> list:
        return []

class ResultCard(QFrame):
    def __init__(self, item: dict, parent=None, main_window=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.main_window = main_window
        self.setup_ui(item)

    def setup_ui(self, item: dict):
        layout = QGridLayout(self)
        layout.setSpacing(12)

        self.setStyleSheet("""
            ResultCard {
                background-color: #1e1e1e;
                border: 2px solid #1a73e8;
                border-radius: 15px;
                padding: 15px;
                margin: 8px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #1a73e8;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4285f4;
            }
        """)

        site_name = item.get('site', {}).get('name', 'نامشخص')
        title = QLabel(f"{item.get('title', '')} ({site_name})")
        title.setFont(QFont('Arial', 12, QFont.Bold))
        title.setStyleSheet("color: #00ff00;")
        title.setWordWrap(True)
        layout.addWidget(title, 0, 0, 1, 2)

        snippet = item.get('snippet', '')
        installment_info = extract_installment_info(snippet)
        price = item.get('price', float('inf'))

        price_info = QVBoxLayout()

        cash_price = QLabel(f"قیمت نقدی: {format_price(price)}")
        cash_price.setStyleSheet("color: #00ff00; font-weight: bold;")
        price_info.addWidget(cash_price)

        if installment_info['is_installment']:
            installment_details = []
            if installment_info['prepayment']:
                installment_details.append(f"پیش پرداخت: {format_price(installment_info['prepayment'])}")
            if installment_info['monthly_payment']:
                installment_details.append(f"قسط ماهانه: {format_price(installment_info['monthly_payment'])}")
            if installment_info['months']:
                installment_details.append(f"مدت اقساط: {installment_info['months']} ماه")
            if installment_info['total_price']:
                installment_details.append(f"قیمت نهایی: {format_price(installment_info['total_price'])}")

            for detail in installment_details:
                label = QLabel(detail)
                label.setStyleSheet("color: #ffd700; font-size: 12px;")
                price_info.addWidget(label)

        layout.addLayout(price_info, 1, 0)

        availability = item.get('availability')
        price = item.get('price', float('inf'))

        availability_info = QVBoxLayout()

        if availability is True and price != float('inf'):
            status_text = "✓ موجود"
            status_color = "#00ff00"
            price_text = f"قیمت: {format_price(price)}"
        elif availability is False:
            status_text = "✗ ناموجود"
            status_color = "#ff0000"
            price_text = "قیمت نامشخص"
        else:
            status_text = "؟ وضعیت نامشخص"
            status_color = "#ffff00"
            price_text = f"قیمت: {format_price(price)}" if price != float('inf') else "قیمت نامشخص"

        status = QLabel(status_text)
        status.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        availability_info.addWidget(status)

        price_label = QLabel(price_text)
        price_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        availability_info.addWidget(price_label)

        layout.addLayout(availability_info, 1, 1)

        category = QLabel(f"🏷️ {item.get('category', 'نامشخص')}")
        category.setStyleSheet("color: #00ff00;")
        layout.addWidget(category, 2, 0)

        source = QLabel(f"🔍 {item.get('source', 'نامشخص')}")
        source.setStyleSheet("color: #00ff00;")
        layout.addWidget(source, 2, 1)

        actions = self.create_action_buttons(item)
        layout.addWidget(actions, 3, 0, 1, 2)

    def create_action_buttons(self, item: dict) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 0)

        for icon, text, action in [
            ("🔗", "مشاهده", lambda: self.main_window.open_link(item['link'])),
            ("📋", "کپی لینک", lambda: self.main_window.copy_to_clipboard(item['link'])),
        ]:
            btn = QPushButton(f"{icon} {text}")
            btn.clicked.connect(action)
            btn.setStyleSheet("""
                QPushButton {
                    background: #2a2a2a;
                    color: #00ff00;
                    border: 1px solid #00ff00;
                    padding: 5px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #3a3a3a;
                    border-color: #00cc00;
                }
            """)
            layout.addWidget(btn)

        return widget

def extract_installment_info(text: str) -> dict:
    installment_info = {
        'is_installment': False,
        'prepayment': 0,
        'monthly_payment': 0,
        'months': 0,
        'total_price': 0
    }

    try:
        if any(term in text.lower() for term in ['قسطی', 'اقساط', 'پیش پرداخت', 'ماهانه']):
            installment_info['is_installment'] = True

            prepayment_pattern = r'(?:پیش پرداخت|پیش‌پرداخت)[:\s]+(\d{1,3}(?:,\d{3})*)'
            prepayment_match = re.search(prepayment_pattern, text)
            if prepayment_match:
                installment_info['prepayment'] = float(prepayment_match.group(1).replace(',', ''))

            monthly_pattern = r'(?:قسط|پرداخت)[:\s]+(?:ماهانه|ماهیانه)[:\s]+(\d{1,3}(?:,\d{3})*)'
            monthly_match = re.search(monthly_pattern, text)
            if monthly_match:
                installment_info['monthly_payment'] = float(monthly_match.group(1).replace(',', ''))

            months_pattern = r'(\d+)\s*(?:ماه|ماهه)'
            months_match = re.search(months_pattern, text)
            if months_match:
                installment_info['months'] = int(months_match.group(1))

            if installment_info['months'] and installment_info['monthly_payment']:
                total = (installment_info['months'] * installment_info['monthly_payment']) + installment_info['prepayment']
                installment_info['total_price'] = total

    except Exception as e:
        logging.error(f"Error extracting installment info: {e}")

    return installment_info

def format_price(price: float) -> str:
    try:
        if price == float('inf') or price == 0:
            return "نامشخص"

        if price > 10_000_000_000:
            price /= 10

        if price >= 1_000_000_000:
            return f"{price / 1_000_000_000:.1f} میلیارد تومان"
        elif price >= 1_000_000:
            return f"{price / 1_000_000:.1f} میلیون تومان"
        elif price >= 1_000:
            return f"{price / 1_000:.1f} هزار تومان"
        else:
            return f"{int(price):,} تومان"
    except:
        return "نامشخص"

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.results_widgets = []
        self.initUI()
        self.feedback_data = []

    def initUI(self):
        self.setWindowTitle('V7lthronyx FairDeal - جستجوی محصولات')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 2px solid #1a73e8;
                padding: 10px;
                border-radius: 10px;
                font-size: 16px;
            }
            QLineEdit:focus {
                border: 2px solid #4285f4;
                background-color: #2d2d2d;
            }
            QPushButton {
                background-color: #1a73e8;
                color: #ffffff;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                min-width: 120px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4285f4;
            }
            QPushButton:disabled {
                background-color: #303030;
                color: #707070;
            }
            QProgressBar {
                border: 2px solid #1a73e8;
                border-radius: 10px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
                height: 30px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a73e8, stop:1 #4285f4);
                border-radius: 8px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 16px;
                border-radius: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #1a73e8;
                min-height: 40px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4285f4;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border-top: 2px solid #1a73e8;
                padding: 8px;
                font-weight: bold;
            }
            QCheckBox {
                spacing: 8px;
                color: #e0e0e0;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
                border: 2px solid #1a73e8;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background-color: #1a73e8;
                image: url(check.png);
            }
            QGroupBox {
                border: 2px solid #1a73e8;
                border-radius: 10px;
                margin-top: 1em;
                padding-top: 1em;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #1a73e8;
                font-weight: bold;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("V7lthronyx FairDeal")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #007acc;
            margin: 20px;
            padding: 10px;
            border-bottom: 3px solid #007acc;
        """)
        main_layout.addWidget(title_label)

        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("نام محصول را وارد کنید...")
        search_layout.addWidget(self.search_input)

        self.search_button = QPushButton("جستجو")
        self.search_button.clicked.connect(self.start_search)
        search_layout.addWidget(self.search_button)

        main_layout.addLayout(search_layout)

        options_group = QGroupBox("گزینه‌ها")
        options_layout = QHBoxLayout()

        self.tor_checkbox = QCheckBox("استفاده از Tor")
        options_layout.addWidget(self.tor_checkbox)

        self.api_checkbox = QCheckBox("استفاده از API")
        self.api_checkbox.setChecked(True)
        options_layout.addWidget(self.api_checkbox)

        self.scrape_checkbox = QCheckBox("استفاده از Scraping")
        self.scrape_checkbox.setChecked(True)
        options_layout.addWidget(self.scrape_checkbox)

        self.selenium_checkbox = QCheckBox("استفاده از Selenium")
        self.selenium_checkbox.setChecked(True)
        options_layout.addWidget(self.selenium_checkbox)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        results_container = QWidget()
        self.results_layout = QVBoxLayout(results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(results_container)
        main_layout.addWidget(scroll_area)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("آماده")
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)

    def start_search(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "هشدار", "لطفاً نام محصول را وارد کنید.")
            return

        self.clear_results()

        search_methods = []
        if self.api_checkbox.isChecked():
            search_methods.append("api")
        if self.scrape_checkbox.isChecked():
            search_methods.append("scrape")
        if self.selenium_checkbox.isChecked():
            search_methods.append("selenium")

        use_tor = self.tor_checkbox.isChecked()

        self.toggle_ui(False)
        self.status_bar.showMessage("در حال جستجو...")

        self.worker = Worker(product_name=query, use_tor=use_tor, search_methods=search_methods)
        self.worker.progress.connect(self.update_progress)
        self.worker.result_ready.connect(self.display_results)
        self.worker.status_message.connect(self.update_status)
        self.worker.start()

    def toggle_ui(self, enabled: bool):
        self.search_input.setEnabled(enabled)
        self.search_button.setEnabled(enabled)
        self.tor_checkbox.setEnabled(enabled)
        self.api_checkbox.setEnabled(enabled)
        self.scrape_checkbox.setEnabled(enabled)
        self.selenium_checkbox.setEnabled(enabled)

    def clear_results(self):
        for widget in self.results_widgets:
            widget.setParent(None)
        self.results_widgets.clear()

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def update_status(self, message: str):
        self.status_bar.showMessage(message)

    def display_results(self, results: dict):
        self.toggle_ui(True)
        self.progress_bar.setValue(100)
        self.status_bar.showMessage("جستجو به پایان رسید.")

        for category, items in results.items():
            if items:
                category_label = QLabel(f"🔖 {category}")
                category_label.setFont(QFont('Arial', 16, QFont.Bold))
                category_label.setStyleSheet("color: #1a73e8;")
                self.results_layout.addWidget(category_label)

                for item in items:
                    card = ResultCard(item, main_window=self)
                    self.results_layout.addWidget(card)
                    self.results_widgets.append(card)

        if not any(results.values()):
            QMessageBox.information(self, "نتیجه", "هیچ نتیجه‌ای یافت نشد.")

    def open_link(self, url: str):
        import webbrowser
        webbrowser.open(url)

    def copy_to_clipboard(self, url: str):
        clipboard = QApplication.clipboard()
        clipboard.setText(url)
        QMessageBox.information(self, "کپی شد", "لینک محصول کپی شد.")

def main():
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
