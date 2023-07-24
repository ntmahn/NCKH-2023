import random
import copy
import numpy as np
import collections
import pandas as pd
import math

number_of_trucks=2
number_of_ATM=10

def readdata(tenfile):
    f=open(tenfile)
    soluongatm= int(f.readline().split()[1])
    f.readline()
    atm=[]
    for i in range (soluongatm):
        phantu=f.readline().split(',')
        atm.append([float(phantu[0]), float(phantu[1])])
    return np.array(atm)
coordinates_matrix=readdata(r'C:\Users\nttgh\Downloads\toado30ATM (1).csv')

# def distance(city1, city2):
#     return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# def distance(lat1, lon1, lat2, lon2):
#     R = 6371  # radius of the Earth i meters
#     d_lat = np.radians(lat2 - lat1)
#     d_lon = np.radians(lon2 - lon1)
#     a = np.sin(d_lat/2) * np.sin(d_lat/2) + np.cos(np.radians(lat1)) \
# * np.cos(np.radians(lat2)) * np.sin(d_lon/2) * np.sin(d_lon/2)
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#     d = R * c
#     return d

# matrankc=[]
# for i in range (len(coordinates_matrix)):
#     kc2diem=[]
#     for j in range (len(coordinates_matrix)):
#         if(i==j):
#             kc2diem.append(0)
#         else:
#             kc2diem.append(distance(coordinates_matrix[i],coordinates_matrix[j])*1000)
#     matrankc.append(kc2diem)
# print(np.array(matrankc))

# coord_matrix = np.array(coordinates_matrix)

# n = coord_matrix.shape[0]
# matrankc = np.zeros((n,n))

# # calculate the distances using Haversine formula
# for i in range(n):
#     for j in range(n):
#         if i != j:
#             matrankc[i,j] = distance(coord_matrix[i,0],coord_matrix[i,1],coord_matrix[j,0],coord_matrix[j,1])
# print(matrankc)

def distance(x1,x2,y1,y2):
    # Đổi đơn vị sang radian
    lat1= (x1 * 0.01746031)
    long1=(y1 * 0.01746031)
    lat2= (x2 * 0.01746031) 
    long2=(y2 * 0.01746031)
    difflong=long2-long1
    return 6378 * math.acos((math.sin(lat1) * math.sin(lat2)) + math.cos(lat1) * math.cos(lat2) * math.cos(difflong))

matrankc=[]
for i in range (len(coordinates_matrix)):
    kc2diem=[]
    for j in range (len(coordinates_matrix)):
        if(i==j):
            kc2diem.append(0)
        else:
            kc2diem.append(distance(coordinates_matrix[i][0],coordinates_matrix[j][0],coordinates_matrix[i][1],coordinates_matrix[j][1]))

    matrankc.append(kc2diem)
print(np.array(matrankc))

def tachdoan(Lotrinh):
    routes=[]
    route=[]
    for i in range(len(Lotrinh)):
        if Lotrinh[i]!=0:
            route.append(Lotrinh[i])
        else:
            routes.append(route)
            route=[]
    routes.append(route)
    return routes

def tongkc(route):
    kc=0
    n=len(route)
    for i in range(n-1):
         kc=kc+matrankc[route[i]][route[i+1]]
    kc=kc+matrankc[0][route[0]]+matrankc[route[n-1]][0]
    return kc

def time_v(starttime,vmax,m):
    #m là thời gian trong 1 khung giờ
    hesotron=[0.5,0.5,0.6,0.7,0.8,0.9,1,0.7,0.8,0.9,0.5,1,1,1,1,1,1]
    current_v=[]
    for i in range (0,len(hesotron),m):
        v=vmax*hesotron[i]
        current_v.append([i+starttime,i+starttime+m,v])
    return np.array(current_v)
Vdc=time_v(7,30,1)

def totaltime (Vdc,totaldistance):
    total_time=0
    current_dis=0
    for i in range (len(Vdc)):
        current_v=Vdc[i][2]
        total_time=total_time+(Vdc[i][1]-Vdc[i][0])
        current_dis+=(Vdc[i][1]-Vdc[i][0])*Vdc[i][2]
        if current_dis>=totaldistance:
            current_dis=current_dis-((Vdc[i][1]-Vdc[i][0])*Vdc[i][2])
            current_dis=totaldistance-current_dis
            total_time=total_time-(Vdc[i][1]-Vdc[i][0])+current_dis/current_v
            break
    return total_time

def Fitness(lotrinh):
    time=[]
    routes=tachdoan(lotrinh)
    r=len(routes)
    for i in range(r):
        if(len(routes[i])==0):
            s=0
        else:
            s=tongkc(routes[i])
            t= totaltime(Vdc,s)
            time.append(t)
    finish_time=max(time)
    return finish_time

def create_individual(n, m):
    Fit=0
    individual = [[],Fit,1]
    for i in range(1,n+m-1):
        individual[0].append (i)
    random.shuffle(individual[0])
    for i in range (len(individual[0])):
        if individual[0][i]>=n: individual[0][i]=0
    individual[1]=Fitness(individual[0])
    return individual

def initialize_population(populationSize, n, m):
    population = []
    for i in range (populationSize):
        population.append (create_individual(n,m))
    return population
Mypop=initialize_population(10,11,2)
print("Quan the la: ")
print(np.array(Mypop))

#Tinh do dai cua hanh trinh cua cac xe, phuc vu cho viec check trung cac loi giai
def len_tour(route):
    temp = copy.copy(route)
    temp.insert(0,0)
    temp.append(0)
    index = []
    array = []
    for i in range(number_of_trucks):
        array.append([])
    for i in range(len(temp)):
        if temp[i] == 0: index.append(i)
    for i in range(number_of_trucks):
        array[i] = index[i+1] - index[i] - 1
    return array

# Kiem tra xem 2 lo trinh co bi trung nhau hay khong
def different(chromosome1, chromosome2):
    x1 = sorted(len_tour(chromosome1[0]))
    x2 = sorted(len_tour(chromosome2[0]))
    if x1 != x2:
        return True
    else:
        array1 = []
        for i in range(number_of_trucks):
            array1.append([])
        index1 = 0
        for i in range(len(chromosome1[0])):
            if chromosome1[0][i] != 0: array1[index1].append(chromosome1[0][i])
            else: index1 = index1 + 1
        array2 = []
        for i in range(number_of_trucks):
            array2.append([])
        index2 = 0
        for i in range(len(chromosome2[0])):
            if chromosome2[0][i] != 0: array2[index2].append(chromosome2[0][i])
            else: index2 = index2 + 1
        array1 = [sorted(element) for element in array1]
        array2 = [sorted(element) for element in array2]
        hashed1 = [hash(tuple(sorted(sub))) for sub in array1]
        hashed2 = [hash(tuple(sorted(sub))) for sub in array2]
        x = collections.Counter(hashed1)
        y = collections.Counter(hashed2)
        if x != y:
            return True
        else:
            return False
        
#Kiem tra 1 lo trinh da ton tai trong quan the hay chua
def exist(population, individual):
    for element in population:
        if different(element, individual) == False:
            return True, element
    return False, individual

 #ham trung gian de xet xem voi 1 diem bat ky thi diem gan no nhat thuoc rout nao
def nearest_city(city, route):
    if len(route) == 1: return route[0]
    else:
        min = max(matrankc[city][route[0]], matrankc[city][route[1]])
        index = 0
        for i in route:
            if matrankc[city][i] <= min and i != city:
                min = matrankc[city][i]
                index = i
    return index

def sorted_route(route):
    fake = copy.copy(route)
    array = []
    route.insert(0, 0)
    point = 0
    for i in range(len(route) - 1):
        point = nearest_city(point, fake)
        array.append(point)
        fake.remove(point)
    return array

#Lua chon ca the 
def roulette_wheel_selection(population):
    total_fitness = 0
    for i in range(len(population)):
        total_fitness = total_fitness + population[i][1]
    probabilities = []
    for i in range(len(population)):
        probabilities.append((total_fitness - population[i][1]))
    probabilities = [p / sum(probabilities) for p in probabilities]
    r = random.random()
    cumulative_probability = 0
    for i in range(len(population)):
        cumulative_probability += probabilities[i]
        if r < cumulative_probability:
            return population[i]
        
def tournament_selection(population, Tournament_size):
    set1 = random.choices(population, k = Tournament_size)
    set1.sort(key = lambda x: x[1])
    parent1 = set1[0]
    set2 = random.choices(population, k = Tournament_size)
    set2.sort(key = lambda x: x[1])
    parent2 = set2[0]
    return parent1, parent2

#Lai ghep 1 diem
def number_of_0(chromosome):
    number = 0
    for i in chromosome:
        if i == 0:
            number = number + 1
    return number 

# Ham lai ghep
def one_point_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [[], 0, 1]
        child2 = [[], 0, 1]
        point = random.randint(0,len(parent1[0]))
        child1[0] = parent1[0][0:point]
        num0 = number_of_0(child1[0])
        for k in parent2[0]:
            if (k in child1[0]) == False and k != 0:
                child1[0].append(k)
            elif k == 0 and num0 < number_of_trucks - 1:
                child1[0].append(k)
                num0 = num0 + 1
        child1[1] = Fitness(child1[0]) 
        child2[0] = parent2[0][0:point]
        num0 = number_of_0(child2[0])
        for k in parent1[0]:
            if (k in child2[0]) == False and k != 0:
                child2[0].append(k)
            elif k == 0 and num0 < number_of_trucks - 1:
                child2[0].append(k)
                num0 = num0 + 1
        child2[1] = Fitness(child2[0]) 
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def two_point_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [[-1] * len(parent1[0]), 0, 1]
        child2 = [[-1] * len(parent2[0]), 0, 1]
        point1 = random.randint(0, len(parent1[0]) - 2)
        point2 = random.randint(point1, len(parent1[0]) - 1)
        child1[0][:point1] = parent1[0][:point1]
        child1[0][point2:] = parent1[0][point2:]
        fake1 = []
        for k in range(len(parent2[0])):
            if (parent2[0][k] in child1[0]) == False and parent2[0][k] != 0:
                fake1.append(parent2[0][k])
            elif parent2[0][k] == 0 and number_of_0(fake1) + number_of_0(child1[0]) != number_of_trucks - 1:
                fake1.append(0)
        child1[0][point1:point2] = fake1
        child1[1] = Fitness(child1[0]) 
        child2[0][:point1] = parent2[0][:point1]
        child2[0][point2:] = parent2[0][point2:]
        fake2 = []
        for k in range(len(parent1[0])):
            if (parent1[0][k] in child2[0]) == False and parent1[0][k] != 0:
                fake2.append(parent1[0][k])
            elif parent1[0][k] == 0 and number_of_0(fake2) + number_of_0(child2[0]) != number_of_trucks - 1:
                fake2.append(0)
        child2[0][point1:point2] = fake2
        child2[1] = Fitness(child2[0]) 
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def crossover(population, parent1, parent2, crossover_rate):
    child1_one, child2_one = one_point_crossover(parent1, parent2, crossover_rate)
    child1_two, child2_two = two_point_crossover(parent1, parent2, crossover_rate)
    temp = exist(population, child1_one)
    if temp[0] == True:
        child1_one[2] = temp[1][2]
        child1_one[1] = Fitness(child1_one[0])
    temp = exist(population, child1_two)
    if temp[0] == True:
        child1_two[2] = temp[1][2]
        child1_two[1] = Fitness(child1_two[0]) 

    temp = exist(population, child2_one)
    if temp[0] == True:
        child2_one[2] = temp[1][2]
        child2_one[1] = Fitness(child2_one[0]) 
    temp = exist(population, child2_two)
    if temp[0] == True:
        child2_two[2] = temp[1][2]
        child2_two[1] = Fitness(child2_two[0]) 

    array1 = [child1_one, child1_two]
    array2 = [child2_one, child2_two]
    array1 = sorted(array1, key=lambda x: x[1])
    array2 = sorted(array2, key=lambda x: x[1])
    child1 = array1[0]
    child2 = array2[0]
    if exist(population, child1)[0] == True:
        child1 = array1[1]
        if exist(population, child1)[0] == True:
            child1 = array1[0]
    if exist(population, child2)[0] == True:
        child2 = array2[1]
        if exist(population, child2)[0] == True:
            child2 = array2[0]
    return child1, child2

def swap_truck_position(solution):
    point1 = random.randint(0, number_of_trucks - 2)
    point2 = random.randint(point1, number_of_trucks - 1)
    array = []
    for i in range(number_of_trucks):
        array.append([])
    index = 0
    for i in range(0, len(solution[0])):
        if solution[0][i] != 0:
            array[index].append(solution[0][i])
        else:
            index = index + 1
    array[point1], array[point2] = array[point2], array[point1]
    new = []
    for i in range(len(array)):
        new.extend(array[i])
        new.append(0)
    new.pop(len(new) - 1)
    new = [new, solution[1], solution[2]]
    return new

def swap_mutation(chromosome, mutation_rate):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0,len(chromosome))
        point2 = random.randint(0,len(chromosome))
        child[0][point1], child[0][point2] = child[0][point2], child[0][point1]
        child = [child[0], Fitness(child[0]), 1]
        return child
    return chromosome

def inversion_mutation(chromosome, mutation_rate):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0, len(child) - 2)
        point2 = random.randint(point1 + 1, len(child) - 1)
        b = child[0][point1 : point2 + 1]
        b.reverse()
        child[0][point1 : point2 + 1] = b
        child = [child[0], Fitness(child[0]),1]
        return child
    return chromosome

def mutation(population, child, mutation_rate):
    child_swap = swap_mutation(child, mutation_rate)
    child_inversion = inversion_mutation(child, mutation_rate)
    temp = exist(population, child_swap)
    if temp[0] == True:
        child_swap[2] = temp[1][2]
        child_swap[1] = Fitness(child_swap[0])
    temp = exist(population, child_inversion)
    if temp[0] == True:
        child_inversion[2] = temp[1][2]
        child_inversion[1] = Fitness(child_inversion[0])
    array = [child_swap, child_inversion]
    array = sorted(array, key=lambda x: x[1])
    child1 = array[0]
    if exist(population, child1)[0] == True:
        child1 = array[1]
        if exist(population, child1)[0] == True:
            child1 = array[0]
    for element in population:
        if different(element, child1) == False:
            element = copy.deepcopy(child1)
    return child1

def Genetic_Algorithm(current_population, tournament_size, crossover_rate, mutation_rate, number_iteration):
    best = [[], pow(10, number_of_trucks)]
    for i in range(2*number_iteration):
        new_population = []
        for j in range(len(current_population)):
            # for k in range(20):
                current_population[j] = swap_truck_position(current_population[j])
        for j in range(int(len(current_population)/2) ):
            #Crossover:
            if i <= number_iteration/2: parent1, parent2 = tournament_selection(current_population, tournament_size)
            else: parent1, parent2 = roulette_wheel_selection(current_population), roulette_wheel_selection(current_population)
            child1, child2 = crossover(new_population, parent1, parent2, crossover_rate)
            #Mutation:
            child1, child2 = mutation(new_population, child1, mutation_rate), mutation(new_population, child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)
        tick = int(len(current_population) * 70/100)
        new_population = sorted(new_population, key = lambda x: x[1])
        new_population1 = []
        for k in new_population:
            if exist(new_population1, k)[0] == False: new_population1.append(k)
            if len(new_population1) == tick: break
        length = len(new_population1)
        current_population = sorted(current_population, key = lambda x: x[1])
        for k in current_population:
            if exist(new_population1, k)[0] == False: new_population1.append(k)
            if len(new_population1) == min(50, number_of_ATM): break
        length = len(new_population1)
        temp = []
        if length != min(50, number_of_ATM):
            # temp = initialize_population(min(50, number_of_ATM) - length + 1)
            temp = initialize_population(10,11,2)
            temp.pop(0)
        new_population1 = new_population1 + temp
        for j in range(len(new_population1)):
            temp = exist(current_population, new_population1[j])
            if temp[0] == True:
                new_population1[j][2] == temp[1][2] + 1
                new_population1[j][1] = Fitness(new_population1[j][0]) 
        current_population = new_population1
        current_population = sorted(current_population, key = lambda x:x[1])
        best = copy.deepcopy(current_population[0])
    return best

k = 5
array = []
for i in range(k):
    solution = Genetic_Algorithm(Mypop, 4, 0.95, 1/min(50, number_of_ATM), 50)
    array.append(solution)
print('Cac lo trinh tot nhat thu duoc sau 5 lan chay:', array)
min=array[0][1]
for i in range(len(array)):
    if array[i][1]<min: min=array[i][1]
for i in range(len(array)):    
 if array[i][1]==min: 
    print('Lo trinh tot nhat la:')
    print(tachdoan(array[i][0]))
    print('Thoi gian doi voi lo trinh tot nhat: ',array[i][1])
    break

# import time
# start=time.time()
# k = 5
# array = []
# for i in range(k):
#     print(i)
#     solution = Genetic_Algorithm((Mypop, 1), 4, 0.95, 1/min(50, number_of_ATM), 50)
#     array.append(solution)
# end=time.time()
# time=end-start
# print ("Compute time:{0}".format(time) + "[sec]")
# best = min(array, key = lambda x: x[1])
# worst = max(array, key = lambda x: x[1])
# sum = 0
# for i in range(k):
#     sum = sum + array[i][1]
# average = sum/k

# temp = 0
# for i in range(len(array)):
#     temp = temp + pow(array[i][1] - average, 2)