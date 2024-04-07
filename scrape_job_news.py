import requests
from bs4 import BeautifulSoup

def get_text_from_top_articles(url, num_articles):
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.find_all("div", class_="views-row")

        top_article_links = []
        for article in articles[:3]:
            link = article.find("a")["href"]
            top_article_links.append(link)

        for link in top_article_links:
            response_article = requests.get(link)
            if response.status_code == 200:
                soup_article = BeautifulSoup(response_article.content, "html.parser")
                text_elements = soup_article.find_all(text=True)
                raw_text = ' '.join(text_elements)

    return raw_text