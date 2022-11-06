import json

data = {"name":"Karan","age":19, "gender": "Male"}

temp = json.dumps(data, indent =3)

print(temp)

# temp_file = json.dumps(data)

# with open("my_file.json","w") as file:
#     file.write(temp_file)

# with open("C:/KM LEE/JSON_Upgrade/my_file.json","r") as file:
#     temp = json.load(file)
    
# print(temp)