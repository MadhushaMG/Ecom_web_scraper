import requests
from bs4 import BeautifulSoup
import json
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk
from io import BytesIO
import re
from datetime import datetime
import pandas as pd
import time
from urllib.parse import urlparse, urljoin
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.ticker as ticker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
products = []
scraped_pages = 0
total_pages = 0
stop_scraping = False
WAIT_TIME = (1, 3)  # Random wait time range in seconds

# AI configuration
MODEL_PATH = "ecommerce_site_classifier.joblib"
FEATURES_PATH = "site_features.joblib"

class EcommerceSiteAnalyzer:
    """AI-powered e-commerce site analyzer for adaptive scraping | Codara"""
    
    def __init__(self):
        """Initialize the analyzer, load model if exists"""
        self.model = None
        self.feature_extractor = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature extractor if they exist"""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.feature_extractor = joblib.load(FEATURES_PATH)
                logger.info("AI model loaded successfully")
            else:
                logger.warning("AI model files not found, will use fallback methods")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
    
    def identify_site_type(self, html_content, url):
        """Identify the type of e-commerce site based on HTML content"""
        try:
            if self.model and self.feature_extractor:
                # Extract features from the HTML
                features = self.extract_features(html_content, url)
                
                # Make prediction
                site_type = self.model.predict([features])[0]
                
                logger.info(f"AI identified site type: {site_type}")
                return site_type
            else:
                # Fallback method: basic pattern recognition
                return self.basic_site_identification(html_content)
        except Exception as e:
            logger.error(f"Error in AI site identification: {e}")
            return self.basic_site_identification(html_content)
    
    def extract_features(self, html_content, url):
        """Extract features from HTML for classification"""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Example features:
        features = {
            'has_woocommerce_class': 1 if soup.select('.woocommerce') else 0,
            'has_shopify_elements': 1 if 'Shopify.theme' in html_content else 0,
            'has_magento_elements': 1 if 'Mage' in html_content or 'magento' in html_content.lower() else 0,
            'url_has_shop': 1 if 'shop' in url.lower() else 0,
            'url_has_product': 1 if 'product' in url.lower() else 0,
            'has_cart_element': 1 if soup.select('.cart') or soup.select('#cart') else 0,
            'has_product_grid': 1 if soup.select('.products') or soup.select('.product-grid') else 0,
        }
        
        # Convert to list in the correct order expected by the model
        feature_list = [features[f] for f in self.feature_extractor]
        return feature_list
    
    def basic_site_identification(self, html_content):
        """Fallback method to identify site type using basic pattern matching"""
        if 'woocommerce' in html_content.lower():
            return 'woocommerce'
        elif 'shopify' in html_content.lower():
            return 'shopify'
        elif 'magento' in html_content.lower():
            return 'magento'
        elif 'opencart' in html_content.lower():
            return 'opencart'
        else:
            return 'generic'
    
    def get_selectors_for_site(self, site_type):
        """Return appropriate CSS selectors based on site type"""
        selectors = {
            'woocommerce': {
                'product_container': "ul.products li, .products .product, li.product",
                'product_name': ".woocommerce-loop-product__title, h2, .product-title",
                'product_price': "span.price, .price, .product-price",
                'regular_price': "del, .regular-price",
                'sale_price': "ins, .sale-price",
                'stock_status': ".stock-status, .stock, .availability",
                'out_of_stock': ".out-of-stock, .outofstock",
                'pagination': ".page-numbers, .pagination, .woocommerce-pagination a",
            },
            'shopify': {
                'product_container': ".product-card, .grid__item, .grid-product",
                'product_name': ".product-card__name, .product-title, .grid-product__title",
                'product_price': ".product-card__price, .product-price, .grid-product__price",
                'regular_price': ".regular-price, .product-price__regular, .grid-product__price--regular",
                'sale_price': ".sale-price, .product-price__sale, .grid-product__price--sale",
                'stock_status': ".product-card__availability, .product-availability",
                'out_of_stock': ".sold-out, .product-price--sold-out",
                'pagination': ".pagination-custom, .pagination, .paginate",
            },
            'magento': {
                'product_container': ".product-items li, .products-grid .item, .products .product-item",
                'product_name': ".product-item-name, .product-name, .product-item-link",
                'product_price': ".price-box, .price-container, .price-wrapper",
                'regular_price': ".old-price, .regular-price",
                'sale_price': ".special-price, .sale-price",
                'stock_status': ".stock, .availability, .product-availability",
                'out_of_stock': ".out-of-stock, .unavailable",
                'pagination': ".pages, .pagination, .toolbar-number",
            },
            'generic': {
                'product_container': ".product, .item, article, .grid-item, [class*='product'], [class*='item']",
                'product_name': "h2, h3, .name, .title, [class*='title'], [class*='name']",
                'product_price': ".price, [class*='price']",
                'regular_price': ".regular-price, .old-price, [class*='regular'], [class*='old']",
                'sale_price': ".sale-price, .special-price, [class*='sale'], [class*='special']",
                'stock_status': ".stock, .availability, .inventory, [class*='stock'], [class*='availability']",
                'out_of_stock': ".out-of-stock, .sold-out, [class*='out-of-stock'], [class*='sold-out']",
                'pagination': ".pagination, .pages, nav, [class*='pagination'], [class*='pages']",
            }
        }
        return selectors.get(site_type, selectors['generic'])

def adaptive_scrape_page(url, session, analyzer, page=1):
    """Scrape a single page of products with adaptive site recognition"""
    global scraped_pages, stop_scraping
    
    if stop_scraping:
        return []

    try:
        if page > 1:
            if '?' in url:
                page_url = f"{url}&paged={page}"
            else:
                page_url = f"{url}?paged={page}"
        else:
            page_url = url
            
        logger.info(f"Requesting page: {page_url}")
        response = session.get(page_url)
        
        # Implement random wait to avoid overloading servers
        wait_time = random.uniform(WAIT_TIME[0], WAIT_TIME[1])
        time.sleep(wait_time)
        
        # Identify site type using AI
        site_type = analyzer.identify_site_type(response.text, page_url)
        selectors = analyzer.get_selectors_for_site(site_type)
        
        soup = BeautifulSoup(response.text, "html.parser")
        page_products = []
        
        # Search for product containers using site-specific selectors
        product_elements = []
        for selector in selectors['product_container'].split(', '):
            elements = soup.select(selector)
            if elements:
                product_elements = elements
                break
        
        logger.info(f"Found {len(product_elements)} product elements on page {page}")
        
        for product in product_elements:
            if stop_scraping:
                return page_products
            
            # Extract product name
            name_elem = None
            for selector in selectors['product_name'].split(', '):
                name_elem = product.select_one(selector)
                if name_elem:
                    break
            name = name_elem.get_text().strip() if name_elem else "No Name"
            
            # Extract price
            price_elem = None
            for selector in selectors['product_price'].split(', '):
                price_elem = product.select_one(selector)
                if price_elem:
                    break
            price = price_elem.get_text().strip() if price_elem else "No Price"
            
            # Extract regular and sale price
            regular_price = "N/A"
            sale_price = "N/A"
            
            if price_elem:
                del_elem = None
                ins_elem = None
                
                for selector in selectors['regular_price'].split(', '):
                    del_elem = price_elem.select_one(selector)
                    if del_elem:
                        break
                        
                for selector in selectors['sale_price'].split(', '):
                    ins_elem = price_elem.select_one(selector)
                    if ins_elem:
                        break
                
                if del_elem and ins_elem:  # On sale
                    regular_price = del_elem.get_text().strip()
                    sale_price = ins_elem.get_text().strip()
                elif del_elem:  # Only regular price shown
                    regular_price = del_elem.get_text().strip()
                elif ins_elem:  # Only sale price shown
                    sale_price = ins_elem.get_text().strip()
                else:  # Only one price shown
                    regular_price = price_elem.get_text().strip()
            
            # Extract stock status
            stock_elem = None
            for selector in selectors['stock_status'].split(', '):
                stock_elem = product.select_one(selector)
                if stock_elem:
                    break
            stock_status = stock_elem.get_text().strip() if stock_elem else "Unknown"
            
            # Check for out of stock indicators if status unknown
            if stock_status == "Unknown":
                out_of_stock_found = False
                for selector in selectors['out_of_stock'].split(', '):
                    if product.select_one(selector) or any(cls in selector for cls in product.get("class", [])):
                        stock_status = "Out of Stock"
                        out_of_stock_found = True
                        break
                
                if not out_of_stock_found:
                    stock_status = "In Stock"  # Assume in stock if not explicitly marked
            
            # Extract image
            image_elem = product.select_one("img")
            image_url = ""
            if image_elem:
                for attr in ["data-src", "data-lazy-src", "src"]:
                    if image_elem.get(attr):
                        image_url = image_elem[attr]
                        # Convert relative URLs to absolute
                        if not image_url.startswith(('http://', 'https://')):
                            image_url = urljoin(page_url, image_url)
                        break
            
            # Get product link
            link_elem = None
            for link_selector in ["a.woocommerce-loop-product__link", "a.product-link", "a.product-title", "a[href]"]:
                link_elem = product.select_one(link_selector)
                if link_elem:
                    break
            
            link = ""
            if link_elem and link_elem.has_attr("href"):
                link = link_elem["href"]
                # Convert relative URLs to absolute
                if not link.startswith(('http://', 'https://')):
                    link = urljoin(page_url, link)
            
            # Extract categories
            categories = []
            
            # Try to extract from breadcrumbs if on product page
            breadcrumbs = soup.select(".woocommerce-breadcrumb a") or soup.select(".breadcrumb a") or soup.select("[class*='breadcrumb'] a")
            if breadcrumbs and len(breadcrumbs) > 1:
                # Skip first (usually Home) and last (current product)
                categories = [crumb.get_text().strip() for crumb in breadcrumbs[1:-1]]
            
            # Add to page products
            product_data = {
                "name": name,
                "price": price,
                "regular_price": regular_price,
                "sale_price": sale_price,
                "stock_status": stock_status,
                "image": image_url,
                "link": link,
                "categories": categories,
                "scrape_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_url": page_url
            }
            
            page_products.append(product_data)
            logger.info(f"Scraped product: {name}")
        
        scraped_pages += 1
        update_progress(scraped_pages, total_pages)
        
        return page_products
    
    except Exception as e:
        logger.error(f"Error scraping page {page}: {e}")
        return []

def adaptive_get_total_pages(url, session, analyzer):
    """Get the total number of pages available using adaptive site recognition"""
    try:
        logger.info(f"Getting total pages for: {url}")
        response = session.get(url)
        site_type = analyzer.identify_site_type(response.text, url)
        selectors = analyzer.get_selectors_for_site(site_type)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try different pagination selectors based on site type
        pagination = None
        for selector in selectors['pagination'].split(', '):
            pagination = soup.select(selector)
            if pagination:
                break
        
        if pagination:
            # Try to find the last page number
            last_page = "1"
            
            # Try method 1: Look for explicit "last page" link
            last_page_elem = soup.select_one(".page-numbers.last, .pagination-link--last, .last")
            if last_page_elem:
                href = last_page_elem.get("href", "")
                match = re.search(r'page=(\d+)', href) or re.search(r'paged=(\d+)', href) or re.search(r'p=(\d+)', href)
                if match:
                    last_page = match.group(1)
            
            # Try method 2: Get the highest numbered pagination link
            if last_page == "1":
                page_nums = []
                for page_link in pagination:
                    if page_link.name == "a" and page_link.get_text().strip().isdigit():
                        page_nums.append(int(page_link.get_text().strip()))
                    elif page_link.name == "span" and page_link.get_text().strip().isdigit():
                        page_nums.append(int(page_link.get_text().strip()))
                
                if page_nums:
                    last_page = str(max(page_nums))
            
            try:
                return int(last_page)
            except:
                return 1
        
        # Check for products count and products per page
        product_count_patterns = [
            r'(\d+)–(\d+) of (\d+)',  # WooCommerce format
            r'Showing (\d+)–(\d+) of (\d+)',  # Another common format
            r'(\d+)-(\d+) of (\d+)',  # Dash instead of en-dash
            r'Items (\d+) to (\d+) of (\d+)'  # Yet another format
        ]
        
        # Look for elements that might contain count information
        result_count_elems = soup.select(".woocommerce-result-count, .showing-count, .results-count, .pagination-result")
        
        for elem in result_count_elems:
            count_text = elem.get_text()
            for pattern in product_count_patterns:
                match = re.search(pattern, count_text)
                if match:
                    per_page = int(match.group(2)) - int(match.group(1)) + 1
                    total = int(match.group(3))
                    return max(1, (total + per_page - 1) // per_page)  # Ceiling division
        
        # Count product items as last resort
        product_elements = []
        for selector in selectors['product_container'].split(', '):
            elements = soup.select(selector)
            if elements:
                product_elements = elements
                break
        
        if product_elements:
            products_per_page = len(product_elements)
            # Look for total count indicators
            total_text = soup.select_one(".total-products, .product-count, [class*='total']")
            if total_text:
                match = re.search(r'(\d+)', total_text.get_text())
                if match:
                    total_products = int(match.group(1))
                    return max(1, (total_products + products_per_page - 1) // products_per_page)
        
        return 1
        
    except Exception as e:
        logger.error(f"Error determining total pages: {e}")
        return 1

def start_adaptive_scraping():
    """Main scraping function that runs in a separate thread with adaptive site recognition"""
    global products, scraped_pages, total_pages, stop_scraping
    
    # Reset status
    products = []
    scraped_pages = 0
    stop_scraping = False
    
    # Clear existing data
    table.delete(*table.get_children())
    status_label.config(text="Starting adaptive scraper...")
    
    # Get user input
    url = url_entry.get().strip()
    if not url:
        url = "https://leyonsuper.com/product-category/sri-lankan-products/"
        url_entry.insert(0, url)
    
    try:
        # Initialize AI analyzer
        analyzer = EcommerceSiteAnalyzer()
        
        # Initialize session with headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
            ]),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Connection": "keep-alive",
        })
        
        # Get total pages
        total_pages = adaptive_get_total_pages(url, session, analyzer)
        status_label.config(text=f"Found {total_pages} pages. Starting adaptive scrape...")
        progress_bar["maximum"] = total_pages
        
        for page in range(1, total_pages + 1):
            if stop_scraping:
                break
                
            status_label.config(text=f"Scraping page {page} of {total_pages}...")
            page_products = adaptive_scrape_page(url, session, analyzer, page)
            products.extend(page_products)
            
            # Update table
            for product in page_products:
                table.insert("", "end", values=(
                    product["name"],
                    product["price"],
                    product["stock_status"],
                    product["link"]
                ))
            
            # Force update GUI
            root.update_idletasks()
        
        if stop_scraping:
            status_label.config(text=f"Scraping stopped. Collected {len(products)} products.")
        else:
            status_label.config(text=f"Scraping complete! Found {len(products)} products.")
        
        # Enable buttons after scrape
        save_csv_btn.config(state="normal")
        save_json_btn.config(state="normal")
        analyze_btn.config(state="normal")
        visualize_btn.config(state="normal")
        stop_btn.config(state="disabled")
        scrape_btn.config(state="normal")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during scraping: {e}")
        status_label.config(text=f"Error: {str(e)}")
        stop_btn.config(state="disabled")
        scrape_btn.config(state="normal")

def update_progress(current, total):
    """Update the progress bar"""
    progress_var.set(current)
    percentage = int((current / total) * 100) if total > 0 else 0
    status_label.config(text=f"Scraped {current} of {total} pages ({percentage}%)...")

def save_as_json():
    """Save scraped data as JSON"""
    if not products:
        messagebox.showwarning("No Data", "No products to save. Please scrape first.")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        initialfile=f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Saved {len(products)} products to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

def save_as_csv():
    """Save scraped data as CSV"""
    if not products:
        messagebox.showwarning("No Data", "No products to save. Please scrape first.")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile=f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    if file_path:
        try:
            # Convert categories list to string for CSV
            products_for_csv = products.copy()
            for product in products_for_csv:
                if isinstance(product['categories'], list):
                    product['categories'] = ', '.join(product['categories'])
            
            df = pd.DataFrame(products_for_csv)
            df.to_csv(file_path, index=False, encoding="utf-8")
            messagebox.showinfo("Success", f"Saved {len(products)} products to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

def show_product_details(event):
    """Show product details when a row is clicked"""
    selected_item = table.selection()
    if not selected_item:
        return
    
    item_values = table.item(selected_item, "values")
    product_name = item_values[0]
    
    # Find the product in our global list
    product = next((p for p in products if p["name"] == product_name), None)
    if not product:
        return
    
    # Update details frame
    product_name_label.config(text=product["name"])
    
    if product["regular_price"] != product["sale_price"] and product["sale_price"] != "N/A":
        price_text = f"Regular: {product['regular_price']}\nSale: {product['sale_price']}"
    else:
        price_text = product["price"]
    
    product_price_label.config(text=price_text)
    product_stock_label.config(text=f"Stock: {product['stock_status']}")
    
    # Load image
    if product["image"]:
        try:
            response = requests.get(product["image"])
            img_data = Image.open(BytesIO(response.content))
            img_data.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img_data)
            product_image_label.config(image=img)
            product_image_label.image = img
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            product_image_label.config(image='')
            product_image_label.config(text="Image not available")
    else:
        product_image_label.config(image='')
        product_image_label.config(text="No image")
    
    # Show product link
    product_link_label.config(text=product["link"])

def extract_numeric_price(price_str):
    """Extract numeric price from string"""
    # Different price patterns
    price_patterns = [
        r'[\$£€₹]?\s?([\d,.]+)',  # Matches currency symbols followed by numbers
        r'([\d,.]+)\s?[\$£€₹]',    # Matches numbers followed by currency symbols
        r'([\d,.]+)',              # Just numbers as fallback
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, price_str)
        if match:
            # Clean the price string (remove commas, etc.)
            price_str = match.group(1).replace(',', '')
            try:
                return float(price_str)
            except:
                pass
    
    return None

def analyze_products():
    """Perform enhanced analysis on scraped products"""
    if not products:
        messagebox.showwarning("No Data", "No products to analyze. Please scrape first.")
        return
    
    try:
        # Convert prices to numeric for analysis
        for product in products:
            # Extract numeric price if possible
            if product["price"] != "No Price":
                product["numeric_price"] = extract_numeric_price(product["price"])
            else:
                product["numeric_price"] = None
            
            # Extract regular and sale prices
            if product["regular_price"] != "N/A":
                product["numeric_regular_price"] = extract_numeric_price(product["regular_price"])
            else:
                product["numeric_regular_price"] = product["numeric_price"]
                
            if product["sale_price"] != "N/A":
                product["numeric_sale_price"] = extract_numeric_price(product["sale_price"])
            else:
                product["numeric_sale_price"] = None
            
        # Count products with prices
        products_with_price = [p for p in products if p["numeric_price"] is not None]
        
        # Calculate basic price stats
        if products_with_price:
            avg_price = sum(p["numeric_price"] for p in products_with_price) / len(products_with_price)
            max_price = max(p["numeric_price"] for p in products_with_price)
            min_price = min(p["numeric_price"] for p in products_with_price)
            
            # Calculate standard deviation
            variance = sum((p["numeric_price"] - avg_price) ** 2 for p in products_with_price) / len(products_with_price)
            std_dev = variance ** 0.5
            
            # Price range counts for histogram
            price_ranges = {}
            range_size = (max_price - min_price) / 5 if max_price > min_price else 1
            for i in range(5):
                lower = min_price + i * range_size
                upper = lower + range_size
                label = f"${lower:.2f} - ${upper:.2f}"
                count = sum(1 for p in products_with_price if lower <= p["numeric_price"] < upper)
                price_ranges[label] = count
            
            # Calculate median price
            sorted_prices = sorted(p["numeric_price"] for p in products_with_price)
            n = len(sorted_prices)
            if n % 2 == 0:
                median_price = (sorted_prices[n//2 - 1] + sorted_prices[n//2]) / 2
            else:
                median_price = sorted_prices[n//2]
        else:
            avg_price = max_price = min_price = median_price = std_dev = 0
            price_ranges = {}
        
        # Count products by stock status
        stock_status_counts = {}
        for p in products:
            status = p["stock_status"].lower()
            if "in stock" in status:
                status = "In Stock"
            elif "out of stock" in status:
                status = "Out of Stock"
            
            if status in stock_status_counts:
                stock_status_counts[status] += 1
            else:
                stock_status_counts[status] = 1
        
        # Calculate discount statistics
        products_on_sale = [p for p in products if p["numeric_regular_price"] is not None 
                           and p["numeric_sale_price"] is not None 
                           and p["numeric_regular_price"] > p["numeric_sale_price"]]
        
        if products_on_sale:
            for p in products_on_sale:
                p["discount_percentage"] = (p["numeric_regular_price"] - p["numeric_sale_price"]) / p["numeric_regular_price"] * 100
            
            avg_discount = sum(p["discount_percentage"] for p in products_on_sale) / len(products_on_sale)
            max_discount = max(p["discount_percentage"] for p in products_on_sale)
            min_discount = min(p["discount_percentage"] for p in products_on_sale)
        else:
            avg_discount = max_discount = min_discount = 0
# Create analysis report dialog
        analysis_dialog = tk.Toplevel(root)
        analysis_dialog.title("Product Analysis Report")
        analysis_dialog.geometry("800x600")
        
        # Create tabs for different analysis views
        tabs = ttk.Notebook(analysis_dialog)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Summary Statistics
        summary_tab = ttk.Frame(tabs)
        tabs.add(summary_tab, text="Summary")
        
        summary_text = scrolledtext.ScrolledText(summary_tab, wrap=tk.WORD)
        summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Build summary report
        summary_content = f"""
        ## Product Analysis Summary
        
        Total Products: {len(products)}
        Products with Price: {len(products_with_price)}
        
        ### Price Statistics
        - Average Price: ${avg_price:.2f}
        - Median Price: ${median_price:.2f}
        - Minimum Price: ${min_price:.2f}
        - Maximum Price: ${max_price:.2f}
        - Price Standard Deviation: ${std_dev:.2f}
        
        ### Stock Status
        {chr(10).join([f"- {status}: {count} products" for status, count in stock_status_counts.items()])}
        
        ### Discount Statistics
        - Products on Sale: {len(products_on_sale)}
        - Average Discount: {avg_discount:.2f}%
        - Minimum Discount: {min_discount:.2f}%
        - Maximum Discount: {max_discount:.2f}%
        """
        
        summary_text.insert(tk.END, summary_content)
        
        # Tab 2: Price Distribution Visualizations
        viz_tab = ttk.Frame(tabs)
        tabs.add(viz_tab, text="Visualizations")
        
        # Create figure and canvas for matplotlib plots
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        fig.subplots_adjust(hspace=0.4)
        
        # Plot 1: Price Histogram
        ax1 = fig.add_subplot(2, 2, 1)
        if price_ranges:
            ax1.bar(price_ranges.keys(), price_ranges.values())
            ax1.set_title("Price Distribution")
            ax1.set_xlabel("Price Range")
            ax1.set_ylabel("Number of Products")
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Stock Status Pie Chart
        ax2 = fig.add_subplot(2, 2, 2)
        if stock_status_counts:
            ax2.pie(stock_status_counts.values(), labels=stock_status_counts.keys(), autopct='%1.1f%%')
            ax2.set_title("Stock Status Distribution")
        
        # Plot 3: Price Boxplot
        ax3 = fig.add_subplot(2, 2, 3)
        if products_with_price:
            ax3.boxplot([p["numeric_price"] for p in products_with_price])
            ax3.set_title("Price Distribution Boxplot")
            ax3.set_ylabel("Price ($)")
            ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))
        
        # Plot 4: Discount Distribution (if applicable)
        ax4 = fig.add_subplot(2, 2, 4)
        if products_on_sale:
            ax4.hist([p["discount_percentage"] for p in products_on_sale], bins=10)
            ax4.set_title("Discount Percentage Distribution")
            ax4.set_xlabel("Discount (%)")
            ax4.set_ylabel("Number of Products")
        
        # Add the plots to the visualization tab
        canvas = FigureCanvasTkAgg(fig, viz_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Raw Data Table
        data_tab = ttk.Frame(tabs)
        tabs.add(data_tab, text="Data Table")
        
        # Create data table
        columns = ("Name", "Regular Price", "Sale Price", "Stock", "Discount %")
        data_table = ttk.Treeview(data_tab, columns=columns, show="headings")
        for col in columns:
            data_table.heading(col, text=col)
            data_table.column(col, width=100)
        
        # Add data to table
        for p in products:
            discount_pct = ""
            if hasattr(p, 'discount_percentage'):
                discount_pct = f"{p['discount_percentage']:.2f}%"
            
            data_table.insert("", "end", values=(
                p["name"],
                p["regular_price"],
                p["sale_price"],
                p["stock_status"],
                discount_pct
            ))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(data_tab, orient="vertical", command=data_table.yview)
        data_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        data_table.pack(fill="both", expand=True)
        
    except Exception as e:
        logger.error(f"Error analyzing products: {e}")
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")

def visualize_data():
    """Create advanced visualizations for the scraped data"""
    if not products:
        messagebox.showwarning("No Data", "No products to visualize. Please scrape first.")
        return
    
    try:
        # Convert all prices to numeric for visualization
        products_with_price = [p for p in products if p.get("numeric_price") is not None]
        
        if not products_with_price:
            messagebox.showwarning("No Price Data", "No products with valid price data to visualize.")
            return
        
        # Create visualization window
        viz_window = tk.Toplevel(root)
        viz_window.title("Advanced Data Visualization")
        viz_window.geometry("1000x800")
        
        # Create notebook for different visualizations
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Price Analysis
        price_tab = ttk.Frame(notebook)
        notebook.add(price_tab, text="Price Analysis")
        
        price_fig = plt.Figure(figsize=(12, 10), dpi=100)
        price_fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Plot 1: Price histogram with KDE
        ax1 = price_fig.add_subplot(2, 2, 1)
        prices = [p["numeric_price"] for p in products_with_price]
        ax1.hist(prices, bins=20, alpha=0.7, density=True)
        
        # Add KDE curve if scipy is imported
        try:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(prices)
            x = np.linspace(min(prices), max(prices), 200)
            ax1.plot(x, density(x), 'r-')
        except:
            pass
        
        ax1.set_title("Price Distribution with Density Curve")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Density")
        
        # Plot 2: Price by stock status boxplot
        ax2 = price_fig.add_subplot(2, 2, 2)
        stock_categories = {}
        for p in products_with_price:
            status = p["stock_status"]
            if status not in stock_categories:
                stock_categories[status] = []
            stock_categories[status].append(p["numeric_price"])
        
        if stock_categories:
            ax2.boxplot(stock_categories.values())
            ax2.set_xticklabels(stock_categories.keys())
            ax2.set_title("Price by Stock Status")
            ax2.set_ylabel("Price")
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Price range counts
        ax3 = price_fig.add_subplot(2, 2, 3)
        price_max = max(prices)
        price_min = min(prices)
        price_range = price_max - price_min
        if price_range > 0:
            bin_count = min(10, len(set(prices)))
            bin_width = price_range / bin_count
            bins = [price_min + i * bin_width for i in range(bin_count + 1)]
            counts, edges = np.histogram(prices, bins=bins)
            bin_labels = [f"${edges[i]:.0f}-${edges[i+1]:.0f}" for i in range(len(edges)-1)]
            ax3.bar(bin_labels, counts)
            ax3.set_title("Products by Price Range")
            ax3.set_xlabel("Price Range")
            ax3.set_ylabel("Number of Products")
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Price clusters (if sklearn is available)
        ax4 = price_fig.add_subplot(2, 2, 4)
        try:
            # Use K-means clustering on prices
            price_data = np.array([p["numeric_price"] for p in products_with_price]).reshape(-1, 1)
            
            # Determine optimal number of clusters using elbow method
            max_clusters = min(6, len(price_data))
            inertia = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(price_data)
                inertia.append(kmeans.inertia_)
            
            # Find elbow point (simple method)
            k_optimal = 2  # Default
            if len(inertia) > 2:
                diffs = np.diff(inertia)
                if any(abs(diffs[:-1]) > abs(diffs[1:])):
                    k_optimal = np.where(abs(diffs[:-1]) > abs(diffs[1:]))[0][0] + 2
            
            # Apply optimal clustering
            kmeans = KMeans(n_clusters=k_optimal, random_state=42)
            clusters = kmeans.fit_predict(price_data)
            
            # Plot clusters
            for i in range(k_optimal):
                cluster_prices = price_data[clusters == i].flatten()
                ax4.hist(cluster_prices, bins=10, alpha=0.5, label=f"Cluster {i+1}")
            
            ax4.set_title(f"Price Clusters (k={k_optimal})")
            ax4.set_xlabel("Price")
            ax4.set_ylabel("Number of Products")
            ax4.legend()
            
        except Exception as clustering_error:
            logger.error(f"Error in clustering: {clustering_error}")
            ax4.text(0.5, 0.5, "Clustering not available", 
                     horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        
        # Add the plots to the price tab
        price_canvas = FigureCanvasTkAgg(price_fig, price_tab)
        price_canvas.draw()
        price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Discount Analysis (if applicable)
        discount_tab = ttk.Frame(notebook)
        notebook.add(discount_tab, text="Discount Analysis")
        
        products_on_sale = [p for p in products_with_price 
                           if p.get("numeric_regular_price") is not None 
                           and p.get("numeric_sale_price") is not None 
                           and p["numeric_regular_price"] > p["numeric_sale_price"]]
        
        if products_on_sale:
            # Calculate discount percentages if not already done
            for p in products_on_sale:
                if "discount_percentage" not in p:
                    p["discount_percentage"] = ((p["numeric_regular_price"] - p["numeric_sale_price"]) 
                                              / p["numeric_regular_price"] * 100)
            
            discount_fig = plt.Figure(figsize=(12, 10), dpi=100)
            discount_fig.subplots_adjust(hspace=0.5, wspace=0.3)
            
            # Plot 1: Discount histogram
            ax1 = discount_fig.add_subplot(2, 2, 1)
            discount_pcts = [p["discount_percentage"] for p in products_on_sale]
            ax1.hist(discount_pcts, bins=10)
            ax1.set_title("Discount Percentage Distribution")
            ax1.set_xlabel("Discount (%)")
            ax1.set_ylabel("Number of Products")
            
            # Plot 2: Scatterplot of original vs sale price
            ax2 = discount_fig.add_subplot(2, 2, 2)
            ax2.scatter([p["numeric_regular_price"] for p in products_on_sale],
                       [p["numeric_sale_price"] for p in products_on_sale], alpha=0.7)
            
            # Add diagonal reference line
            price_max = max([p["numeric_regular_price"] for p in products_on_sale])
            ax2.plot([0, price_max], [0, price_max], 'r--')
            
            ax2.set_title("Regular vs Sale Price")
            ax2.set_xlabel("Regular Price ($)")
            ax2.set_ylabel("Sale Price ($)")
            
            # Plot 3: Discount percentage vs original price
            ax3 = discount_fig.add_subplot(2, 2, 3)
            ax3.scatter([p["numeric_regular_price"] for p in products_on_sale],
                       [p["discount_percentage"] for p in products_on_sale], alpha=0.7)
            ax3.set_title("Discount Percentage vs Regular Price")
            ax3.set_xlabel("Regular Price ($)")
            ax3.set_ylabel("Discount Percentage (%)")
            
            # Plot 4: Discount vs regular price bins
            ax4 = discount_fig.add_subplot(2, 2, 4)
            
            # Create price bins
            regular_prices = [p["numeric_regular_price"] for p in products_on_sale]
            price_min, price_max = min(regular_prices), max(regular_prices)
            bin_count = min(5, len(set(regular_prices)))
            bin_width = (price_max - price_min) / bin_count
            
            avg_discounts = []
            bin_labels = []
            
            for i in range(bin_count):
                bin_lower = price_min + i * bin_width
                bin_upper = bin_lower + bin_width
                bin_products = [p for p in products_on_sale 
                               if bin_lower <= p["numeric_regular_price"] < bin_upper]
                
                if bin_products:
                    avg_discount = sum(p["discount_percentage"] for p in bin_products) / len(bin_products)
                    avg_discounts.append(avg_discount)
                    bin_labels.append(f"${bin_lower:.0f}-${bin_upper:.0f}")
            
            if avg_discounts:
                ax4.bar(bin_labels, avg_discounts)
                ax4.set_title("Average Discount by Price Range")
                ax4.set_xlabel("Regular Price Range")
                ax4.set_ylabel("Average Discount (%)")
                ax4.tick_params(axis='x', rotation=45)
            
            # Add the plots to the discount tab
            discount_canvas = FigureCanvasTkAgg(discount_fig, discount_tab)
            discount_canvas.draw()
            discount_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Show message if no discounted products
            no_discount_label = ttk.Label(discount_tab, text="No products with discount information available")
            no_discount_label.pack(pady=20)
        
        # Tab 3: Data Exploration
        explore_tab = ttk.Frame(notebook)
        notebook.add(explore_tab, text="Data Explorer")
        
        # Add controls for exploring the data
        control_frame = ttk.Frame(explore_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select visualization:").pack(side="left", padx=5)
        
        # Visualization options
        viz_options = ["Price Distribution", "Stock Analysis", "Category Analysis", "Discount Analysis"]
        viz_var = tk.StringVar(value=viz_options[0])
        viz_dropdown = ttk.Combobox(control_frame, textvariable=viz_var, values=viz_options, state="readonly")
        viz_dropdown.pack(side="left", padx=5)
        
        # Frame for the dynamic visualization
        dynamic_frame = ttk.Frame(explore_tab)
        dynamic_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        current_canvas = None
        
        def update_visualization(*args):
            nonlocal current_canvas
            
            # Clear current visualization
            for widget in dynamic_frame.winfo_children():
                widget.destroy()
            
            # Create new figure
            fig = plt.Figure(figsize=(10, 8), dpi=100)
            
            selected_viz = viz_var.get()
            
            if selected_viz == "Price Distribution":
                ax = fig.add_subplot(111)
                prices = [p["numeric_price"] for p in products_with_price]
                ax.hist(prices, bins=20)
                ax.set_title("Price Distribution")
                ax.set_xlabel("Price ($)")
                ax.set_ylabel("Number of Products")
                
            elif selected_viz == "Stock Analysis":
                ax = fig.add_subplot(111)
                
                stock_counts = {}
                for p in products:
                    status = p["stock_status"]
                    stock_counts[status] = stock_counts.get(status, 0) + 1
                
                ax.pie(stock_counts.values(), labels=stock_counts.keys(), autopct='%1.1f%%')
                ax.set_title("Stock Status Distribution")
                
            elif selected_viz == "Category Analysis":
                ax = fig.add_subplot(111)
                
                # Extract all categories
                all_categories = []
                for p in products:
                    if isinstance(p.get("categories"), list):
                        all_categories.extend(p["categories"])
                    elif isinstance(p.get("categories"), str) and p["categories"]:
                        all_categories.extend([cat.strip() for cat in p["categories"].split(',')])
                
                # Count categories
                category_counts = {}
                for cat in all_categories:
                    if cat:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Sort by frequency and take top 10
                top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                if top_categories:
                    ax.bar([cat[0] for cat in top_categories], [cat[1] for cat in top_categories])
                    ax.set_title("Top 10 Categories")
                    ax.set_xlabel("Category")
                    ax.set_ylabel("Number of Products")
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, "No category data available", 
                           horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                
            elif selected_viz == "Discount Analysis":
                ax = fig.add_subplot(111)
                
                if products_on_sale:
                    discount_pcts = [p["discount_percentage"] for p in products_on_sale]
                    ax.hist(discount_pcts, bins=10)
                    ax.set_title("Discount Percentage Distribution")
                    ax.set_xlabel("Discount (%)")
                    ax.set_ylabel("Number of Products")
                else:
                    ax.text(0.5, 0.5, "No products with discount information", 
                           horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            
            # Create canvas with the figure
            canvas = FigureCanvasTkAgg(fig, dynamic_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            current_canvas = canvas
        
        # Connect the dropdown to the update function
        viz_var.trace("w", update_visualization)
        
        # Initialize with the default visualization
        update_visualization()
        
    except Exception as e:
        logger.error(f"Error visualizing data: {e}")
        messagebox.showerror("Visualization Error", f"An error occurred during visualization: {e}")

def stop_scraping_process():
    """Stop the scraping process"""
    global stop_scraping
    stop_scraping = True
    status_label.config(text="Stopping scraper, please wait...")
    save_csv_btn.config(state="normal")
    save_json_btn.config(state="normal")
    scrape_btn.config(state="normal")
    stop_btn.config(state="disabled")

def start_scrape_thread():
    """Start the scraping process in a separate thread"""
    global scrape_thread
    
    url_text = url_entry.get().strip()
    if not url_text:
        messagebox.showwarning("Missing URL", "Please enter a URL to scrape")
        return
    
    # Validate URL
    try:
        result = urlparse(url_text)
        if not all([result.scheme, result.netloc]):
            messagebox.showwarning("Invalid URL", "Please enter a valid URL including http:// or https://")
            return
    except:
        messagebox.showwarning("Invalid URL", "Please enter a valid URL")
        return
    
    # Disable buttons during scrape
    scrape_btn.config(state="disabled")
    save_csv_btn.config(state="disabled")
    save_json_btn.config(state="disabled")
    analyze_btn.config(state="disabled")
    visualize_btn.config(state="disabled")
    stop_btn.config(state="normal")
    
    # Start scraping in a separate thread
    scrape_thread = threading.Thread(target=start_adaptive_scraping)
    scrape_thread.daemon = True
    scrape_thread.start()

# Create main window
root = tk.Tk()
root.title("AI-Powered E-commerce Web Scraper")
root.geometry("1200x800")

# Create main container frame
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

# URL Entry section
url_frame = ttk.Frame(main_frame)
url_frame.pack(fill="x", pady=10)

ttk.Label(url_frame, text="E-commerce Website URL:").pack(side="left", padx=5)
url_entry = ttk.Entry(url_frame, width=80)
url_entry.pack(side="left", padx=5, fill="x", expand=True)

scrape_btn = ttk.Button(url_frame, text="Start Scraping", command=start_scrape_thread)
scrape_btn.pack(side="left", padx=5)

stop_btn = ttk.Button(url_frame, text="Stop", command=stop_scraping_process, state="disabled")
stop_btn.pack(side="left", padx=5)

# Progress section
progress_frame = ttk.Frame(main_frame)
progress_frame.pack(fill="x", pady=10)

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate", variable=progress_var)
progress_bar.pack(side="left", padx=5, fill="x", expand=True)

status_label = ttk.Label(progress_frame, text="Ready to scrape...")
status_label.pack(side="left", padx=5)

# Split main area into two panes
paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
paned_window.pack(fill="both", expand=True, pady=10)

# Left pane: Product table
table_frame = ttk.Frame(paned_window)
paned_window.add(table_frame, weight=2)

# Create treeview for products
columns = ("Name", "Price", "Stock Status", "Link")
table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")

# Set column headings
for col in columns:
    table.heading(col, text=col)
    table.column(col, width=100)  # Default width

# Adjust column widths
table.column("Name", width=250)
table.column("Link", width=300)

# Add scrollbars
y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
table.configure(yscrollcommand=y_scrollbar.set)
y_scrollbar.pack(side="right", fill="y")

x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
table.configure(xscrollcommand=x_scrollbar.set)
x_scrollbar.pack(side="bottom", fill="x")

table.pack(fill="both", expand=True)

# Bind click event to show details
table.bind("<ButtonRelease-1>", show_product_details)

# Right pane: Product details
details_frame = ttk.Frame(paned_window)
paned_window.add(details_frame, weight=1)

# Product details layout
details_frame.columnconfigure(0, weight=1)

product_name_label = ttk.Label(details_frame, text="", font=("Helvetica", 12, "bold"), wraplength=350)
product_name_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)

product_image_label = ttk.Label(details_frame, text="No image selected")
product_image_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)

product_price_label = ttk.Label(details_frame, text="")
product_price_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

product_stock_label = ttk.Label(details_frame, text="")
product_stock_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

link_frame = ttk.Frame(details_frame)
link_frame.grid(row=4, column=0, sticky="w", padx=10, pady=5)

ttk.Label(link_frame, text="Product Link:").pack(side="left")
product_link_label = ttk.Label(link_frame, text="", foreground="blue", cursor="hand2")
product_link_label.pack(side="left", padx=5)

# Bottom buttons section
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill="x", pady=10)

save_csv_btn = ttk.Button(button_frame, text="Save to CSV", command=save_as_csv, state="disabled")
save_csv_btn.pack(side="left", padx=5)

save_json_btn = ttk.Button(button_frame, text="Save to JSON", command=save_as_json, state="disabled")
save_json_btn.pack(side="left", padx=5)

analyze_btn = ttk.Button(button_frame, text="Analyze Products", command=analyze_products, state="disabled")
analyze_btn.pack(side="left", padx=5)

visualize_btn = ttk.Button(button_frame, text="Visualize Data", command=visualize_data, state="disabled")
visualize_btn.pack(side="left", padx=5)

# Initialize the analyzer
analyzer = EcommerceSiteAnalyzer()

# Start the GUI main loop
root.mainloop()
