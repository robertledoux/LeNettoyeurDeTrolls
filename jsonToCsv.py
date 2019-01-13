import json

with open("comments.json", encoding="utf8") as f:
	data = json.load(f)


myData = []
csvfile = open("comments.csv", "a", encoding="utf-8")
for d in data:
	if len(str(d["commentText"])) > 10 and d["numberOfReplies"] is not None:
		print(str(d["numberOfReplies"]))
		ligne = d["commentText"].replace("@"," ").replace("\n"," ").replace("\r"," ").replace(";", "|") + "@" + str(d["numberOfReplies"]) + "@" + str(d["likes"]) + "@" + str(0) + "\n"
		myData.append(ligne.encode("utf-8"))
		#print(ligne)
		csvfile.write(ligne)




