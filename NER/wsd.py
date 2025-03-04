import requests 
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

#-------------------------------------------------------------
# Path (lowest common ancestor) similarity between two synsets
#-------------------------------------------------------------
lakeside = wn.synsets("lakeside", pos="n")[0]
synsets = wn.synsets("bank", pos="n")
print("--------------------------------------------------")
print("Similarity of each meaning of 'bank' to 'lakeside'")
print("--------------------------------------------------")
for s in synsets:
    print(s.definition())
    print(s.path_similarity(lakeside))

#-------------------------------------------------------------
# Querying an entity from Wikidata
#-------------------------------------------------------------
def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was an error'


query = 'Tristan und Isolde'
params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': query,
        'language': 'en',
        'uselang': 'fr'
    }
 
data = fetch_wikidata(params).json()
dataDict = dict(data)

print("-----------------------------------------------------")
print("Search results for 'Tristan und Isolde' from Wikidata")
print("-----------------------------------------------------")

print("\nGlobal result set:")
print(dataDict)
print("\nIndividual results:")
for result in dataDict["search"]:
    print("=======> " + str(result))

identifier = dataDict["search"][0]["id"]

print("-------------------------------------------------------------------")
print("Retrieving all attributes and relations for the first search result")
print("-------------------------------------------------------------------")

params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': identifier,
        'languages': 'en'
    }

entities = fetch_wikidata(params).json()
eDict = dict(entities)
#print(eDict)
