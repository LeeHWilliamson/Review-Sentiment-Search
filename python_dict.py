my_dict = {
    "key1": "value1",
    "key2": "value2"
}
hello_str = "hello"
for key, value in my_dict.items():
    my_dict[key] = set()
    my_dict[key].add(hello_str)
    print(my_dict[key])
    #my_dict[key] = value.replace("'", "")

for key, value in my_dict.items():
    print(value)
    if value == {'hello'}:
        print("key has been found")

#print(type(my_dict["key1"]))
#print(my_dict["key1"])