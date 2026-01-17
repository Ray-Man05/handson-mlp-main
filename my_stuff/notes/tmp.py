names = """
Kaggle.com
UC Irvine Machine Learning Repository
Stanford Large Network Dataset Collection
Amazon's AWS datasets
U.S. Government's Open Data
DataPortals.org
Wikipedia's list of machine learning datasets
"""

links = """
https://kaggle.com/datasets
https://archive.ics.uci.edu
https://snap.stanford.edu/data
https://registry.opendata.aws
https://data.gov
https://dataportals.org
https://homl.info/9
"""

links = links.split("\n")

for i, n in enumerate(names.split("\n")):
    if n and links[i]:
        print(f"- [{n}]({links[i]})")
