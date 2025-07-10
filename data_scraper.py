import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL for per-game stats
url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Find the table by ID
table = soup.find("table", {"id": "per_game_stats"})

# Extract headers
headers = [th.getText() for th in table.find("thead").findAll("th")][1:]  # skip rank
rows = table.find("tbody").findAll("tr")

# Parse player rows
player_stats = []
for row in rows:
    if row.find("th", {"scope": "row"}) is None:
        continue  # skip separator rows
    stats = [td.getText() for td in row.findAll("td")]
    if stats:
        player_stats.append(stats)

# Create DataFrame
df = pd.DataFrame(player_stats, columns=headers)

# Preview
print(df.head())
df.to_csv("nba_per_game_2024.csv", index=False)