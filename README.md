Product Scraper - Leyon Super
This Python-based product scraper allows you to scrape product information from the website Leyon Super and display it in a graphical interface using Tkinter. The tool fetches product names, prices, images, and links, and lets you save the scraped data in a JSON format.

🛠️ Features
Scrape product name, price, image, and link from the Leyon Super website.
View the scraped products in a table within the Tkinter GUI.
Display product images when a row is selected.
Save scraped data in JSON format.
Handle errors gracefully with informative error messages.
Works with a user-friendly interface for easy interaction.
📦 Requirements
To run this application, you'll need the following Python libraries:

requests - for fetching data from the website.


BeautifulSoup4 - for parsing the HTML content.
tkinter - for creating the GUI interface.
Pillow - for displaying product images.
You can install the required libraries using pip:

bash
Copy
Edit
pip install requests beautifulsoup4 pillow
⚙️ How to Use
Run the Script:
Simply execute the Python script product_scraper.py.

bash
Copy
Edit
python product_scraper.py
Scrape Products:

Click the Scrape Products button in the application.
The tool will fetch product details from Leyon Super and display them in the table.
View Products:

You can see the product name, price, and link displayed in the table.
Click on any product row to view its image.
Save Data:

To save the scraped data, click the Download JSON button.
A file dialog will open, allowing you to select a location and save the data in JSON format.

🔧 How It Works
1. Fetching Product Data
The scraper sends an HTTP GET request to Leyon Super's product page.
The HTML content is parsed using BeautifulSoup to extract product details.
Each product's name, price, image URL, and product page link are gathered and stored.


2. Displaying Data in Tkinter GUI
A Tkinter Treeview widget is used to display the product data in a table.
A Pillow image display area is used to show the product image when a row is selected.
3. Saving Data
The scraped data is saved as a JSON file, which includes product details such as name, price, image URL, and product link.
📝 How to Modify or Extend
You can update the URL in the script to scrape products from other websites.
Modify the HTML element selectors (such as class names) if the website structure changes.
Enhance the user interface to show additional product details or add more features.
📄 License
This project is open-source and available under the MIT License.

📞 Contact
If you have any questions or issues, feel free to contact me at:

Email: madhushamalsara@gmail.com
GitHub: https://github.com/@MadhushaMG
