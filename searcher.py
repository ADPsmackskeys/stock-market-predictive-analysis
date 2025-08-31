try:
	from googlesearch import search
except ImportError: 
	print("No module named 'google' found")

# to search
query = "site:moneycontrol.com inurl:stocks-to-watch"
searchlist = []

for j in search (query, num_results = 1000, region = "in", lang = "en"):
	searchlist.append (j)


