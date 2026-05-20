import urllib.request
import urllib.parse
import json

def test():
    # En chainyo/rvl-cdip, filtramos por la columna "label" (14 = scientific_report, 15 = specification)
    where_clause = '"label" = 14'
    encoded_where = urllib.parse.quote(where_clause)
    
    url = f"https://datasets-server.huggingface.co/filter?dataset=chainyo/rvl-cdip&config=default&split=test&where={encoded_where}&offset=0&limit=1"
    print("URL:", url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req).read()
        data = json.loads(res)
        print("Success! Row keys:", data['rows'][0]['row'].keys())
        print("Label of first row:", data['rows'][0]['row']['label'])
    except Exception as e:
        print("Error:", e)
        if hasattr(e, 'read'):
            print("Error body:", e.read().decode('utf-8'))

if __name__ == "__main__":
    test()
