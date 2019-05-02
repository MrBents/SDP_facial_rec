from numpy.random import choice
import pandas as pd


def normalize(arr):
    m = sum(arr)
    return [float(e) / m for e in arr]


def grade_food(foods, age, gender, eth):
    l = []
    for food in foods:
        p = 0.1
        if "chicken" in food.lower():
            p += 1
        if "nuggets" in food.lower():
            p += 0.6
            if age < 12:
                p += 0.4
        if "sandwich" in food.lower():
            p += 0.8
        if "salad" in food.lower():
            p += 0.2
            if gender == "female":
                p += 0.2
            if eth == "indian":
                p += 0.4
            if age < 12:
                p -= 0.2
        if "deluxe" in food.lower():
            p -= 0.4
        if "soup" in food.lower():
            p -= 0.7
        l.append(p)
    return normalize(l)


food = [
    "Spicy Chicken Sandwich",
    "Chicken Sandwich",
    "Chiken Nuggets 10pc",
    "Chiken Nuggets 6pc",
    "Salad Meal",
    "Grilled Chicken Sandwich",
    "Spicy Deluxe Sandwich",
    "chicken noodle soup",
    "chicken tortilla soup",
    "nothing",
]


sides = [
    "Hashbrowns",
    "Fruit Cup",
    # "Hashbrown scramble burrito",
    # "Hashbrown scramble bowl",
    "side salad",
    "waffle fries",
    "superfood side",
    "nothing",
]


def grade_side(sides, age, gender, ethnic):
    l = []
    for side in sides:
        p = 0.3
        if "fries" in side.lower():
            if age < 12:
                p += 0.3
            p += 2
        if "hashbrown" in side.lower():
            p += 1
        if "nothing" in side.lower():
            if age > 40:
                p += 0.4
            p += 0.5
        l.append(p)
    return normalize(l)


drink = ["lemonade", "Milkshake", "soda", "water", "nothing", "tea"]


def grade_drink(age, gender, eth):
    if gender == "male":
        if age < 15:
            return normalize([4, 7, 8, 3, 2, 1])
        return normalize([3, 5, 7, 5, 3, 3])

    else:
        if age < 15:
            return normalize([6, 9, 5, 4, 2, 6])
        return normalize([4, 10, 3, 6, 2, 8])


def grade_age(age):
    if age < 5:
        return 1
    elif age < 10:
        return 2
    elif age < 15:
        return 4
    elif age < 20:
        return 9
    elif age < 35:
        return 20
    elif age < 45:
        return 8
    elif age < 65:
        return 3
    else:
        return 0.5


ages = range(100)
age_p = normalize([grade_age(e) for e in ages])


ethnic = ["white", "black", "india", "asia"]
eth_mapper = {}
for i, e in enumerate(ethnic):
    eth_mapper[e] = i
ethnic_p = normalize([18, 8, 5, 5])

gender = ["male", "female"]
gender_p = [0.6, 0.4]


lista = [["age", "gender", "ethnic_origin", "food", "side", "drink"]]
for i in range(1000):
    i_age = choice(ages, p=age_p)
    i_gender = choice(gender, p=gender_p)
    i_ethnic = choice(ethnic, p=ethnic_p)
    i_food = choice(food, p=grade_food(food, i_age, i_gender, i_ethnic))
    i_sides = choice(sides, p=grade_side(sides, i_age, i_gender, i_ethnic))
    i_drink = choice(drink, p=grade_drink(i_age, i_gender, i_ethnic))

    lista.append([i_age, i_gender, i_ethnic, i_food, i_sides, i_drink])

df = pd.DataFrame(lista[1:], columns=lista[0])
df[["age", "gender", "ethnic_origin", "drink"]].to_csv("drink.csv", index=False)
df[["age", "gender", "ethnic_origin", "food"]].to_csv("food.csv", index=False)
df[["age", "gender", "ethnic_origin", "side"]].to_csv("side.csv", index=False)

# def longestSequence(node, maxi=float('-inf'), cur_count=0):
#         maxx = maxi
#     if cur_count > maxi:
#         maxx = cur_count
#     if node.right != None and node.left != None:
#         return max(longestSequence(node.right, maxi=maxx, cur_count=cur_count + 1), longestSequence(node.left, maxi=maxx))
#     elif node.right != None:
#         return longestSequence(node.right, maxi=maxx, cur_count=cur_count + 1)
#     elif node.left != None:
#         return longestSequence(node.left, maxi=maxx)
#     else:
#         return maxx + 1


#  def longestSequence(node):
#     if node.right != None and node.left != None:
#         return max(longestSequence(node.right) + 1, longestSequence(node.left))
#     elif node.right != None:
#         return longestSequence(node.right) + 1
#     elif node.left != None:
#         return longestSequence(node.left, maxi=maxx)
#     else:
#         return 1
