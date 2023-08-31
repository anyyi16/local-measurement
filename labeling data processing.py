import pandas as pd
import json
with open("C:/Users/Betsy/Desktop/\ds project/analytics/fishLabeling.json", "r") as json_file:
    data = json.load(json_file)
#print(data)
index_number = []
for item in data["frames"]:
    #print(item)
    if "index" in item:
        index_number.append(item["index"])
#print(index_number)
#df = pd.DataFrame({'frame_index': index_number})
#print(df)
dict_labelling = {}
temp_item = None
for index in index_number:
    for item in data["frames"]:
        
        if "index" in item and item["index"] == index:
            temp_item = item["figures"]
            temp_dict = {}
            for figure in temp_item:
                
                object_id = figure.get("objectId")
                #print(object_id)
                origin = figure.get("geometry", {}).get("bitmap", {}).get("origin", [])
                temp_dict[object_id] = origin
            dict_labelling[index] = temp_dict
        
#print(dict_labelling)
df = pd.DataFrame(columns=["frame_index", "fish_index", "x","y"])


for index, nested_dict in dict_labelling.items():
    #print("Index:", index)
    for nested_key in nested_dict:
        #print("Nested key:", nested_key)
        #print("Nested value:", nested_dict[nested_key])
        if len(nested_dict)>20:
            new_row = {"frame_index": index, "fish_index": nested_key, "x": nested_dict[nested_key][0], "y": nested_dict[nested_key][1]}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
outputpath = "C:/Users/Betsy/Desktop/\ds project/analytics/fishLabeling.csv"
df.to_csv(outputpath, sep = ',', index = True, header = True)



