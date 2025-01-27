person = {
    "name": "Siddharth",
    "age": 21,
    "city": "Meerut"
}
print("Name:", person["name"])
print("Age:", person["age"])
print("City:", person["city"])
person["email"] = "Sid.agl77@gmail.com"
print("Updated dictionary with email:", person)
person["age"] = 22
print("Updated dictionary with modified age:", person)
if "city" in person:
    print("Key 'city' exists in the dictionary")

keys = person.keys()
values = person.values()
print("Keys:", list(keys))
print("Values:", list(values))
