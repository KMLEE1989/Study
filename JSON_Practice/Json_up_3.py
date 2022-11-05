student = {'name': "John", 'age':25, 'courses': ['math', 'CompSci']}

# print(student['courses'])

# print(student.get('name'))
# student['phone'] = '555-5555'
# student['name'] = 'Jane'
# student.update({'name': 'Jane', 'age':26, 'phone':'555-5555'})

# # print(student.get('phone','Not Found'))

# print(student)

# del student['age']

# print(student)

# age = student.pop('age')

# # print(student)
# print(age)

# print(len(student))

# print(student.keys())
# print(student.values())

# print(student.items())

for key, value in student.items():
    print(key, value)