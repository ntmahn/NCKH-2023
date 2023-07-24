import numpy
import copy
import math
import random

#đọc tọa độ
def readdata(tenfile):
    f=open(tenfile)
    soluongatm= int(f.readline().split()[1])
    f.readline()
    atm=[]
    for i in range (soluongatm):
        phantu=f.readline().split(',')
        atm.append([float(phantu[0]), float(phantu[1])])
    return numpy.array(atm)
coordinates_matrix=readdata(r'C:\Users\Admin\Downloads\toado20ATM.csv')
print(coordinates_matrix)
#print(len(coordinates_matrix))

#công thức tính khoảng cách
#cho tọa độ 2 điểm, tính ra được khoảng cách 
def khoangcach(D1,D2):
    kc=float(math.sqrt((D1[0]-D2[0])**2+(D1[1]-D2[1])**2))
    return kc

#in ma trận khoảng cách. tỉ lệ bản đồ 1:1000
distance_matrix=[]
for i in range (len(coordinates_matrix)):
    kc2diem=[]
    for j in range (len(coordinates_matrix)):
        if(i==j):
            kc2diem.append(0)
        else:
            kc2diem.append(khoangcach(coordinates_matrix[i],coordinates_matrix[j])*1000)

    distance_matrix.append(kc2diem)
print(numpy.array(distance_matrix))


#tạo ra một lộ trình ngẫu nhiên.
#cho số lượng cây ATM, tạo ra lộ trình ngẫu nhiên.
#Lộ trình có điểm đầu và điểm cuối là 0 (depot)
def create_random_route(number_customer):
    route=[]
    route=random.sample(range(1,number_customer+1),number_customer)
    
    new_route=[]
    new_route.append(0)
    new_route.extend(route)
    new_route.append(0)

    return route,new_route

#tách lộ trình vừa tìm thành n lộ trình cho n xe
#n=số xe
def split(randomroute,n):
    routes = []
    route = []
    route1=[]
    for i in range (len(randomroute)): 
        if (randomroute[i] <n):
            route.append(randomroute[i])
        else:
            routes.append(route)
            route1=route
            route = []
    routes.append(route)
    return routes,route1,route

#tính tổng độ dài lộ trình
def total_distance(route):
    kc=0
    n=len(route)
    for i in range(n-1):
        kc=kc+distance_matrix[route[i]][route[i+1]]
    kc=kc+distance_matrix[0][route[0]]+distance_matrix[route[n-1]][0]
    return kc

#insertion
#tìm lộ trình tối ưu bằng cách đổi chỗ các cây ATM trong lộ trình
def insertion(route):
    dis_min=total_distance(route)
    best_route=route
    for i in range (len(route)):
        for j in range (len(route)):
            if i<j:
                route_tamthoi=[]
                route_tamthoi.extend(route[:i])
                route_tamthoi.extend(route[i+1:j+1])
                route_tamthoi.append(route[i])
                route_tamthoi.extend(route[j+1:])
                new_distance=total_distance(route_tamthoi)
                
            elif i>j:
                route_tamthoi=[]
                route_tamthoi.extend(route[:j+1])
                route_tamthoi.append(route[i])
                route_tamthoi.extend(route[j+1:i])
                route_tamthoi.extend(route[i+1:])
                new_distance=total_distance(route_tamthoi)
            else:
                route_tamthoi=route
                new_distance=total_distance(route_tamthoi)
            if new_distance<dis_min:
                dis_min=new_distance
                best_route=route_tamthoi
    return best_route,dis_min



#thêm vận tốc thay đổi theo khung thời gian
#cho thời gian bắt đầu tiếp quỹ(starttime)
#m là thời gian trong 1 khung giờ
#cho ra kết quả là một mảng gồm nhiều khung giờ [giờ bắt đầu, giờ kết thúc, v di chuyển]
def time_v(starttime,vmax,m):
    hesotron=[0.5,0.5,0.6,0.7,0.8,0.9,1,0.7,0.8,0.9,0.5,1,1,1,1,1,1]
    current_v=[]
    for i in range (0,len(hesotron),m):
        v=vmax*hesotron[i]
        current_v.append([i+starttime,i+starttime+m,v])
    return numpy.array(current_v)

Vdc=time_v(7,30,1)
print(numpy.array(Vdc))

#tính tổng thời gian
#đầu vào: khung vận tốc thay đổi theo thời gian, tổng độ dài lộ trình
#đầu ra: tổng thời gian di chuyển trong lộ trình đó
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

#tối ưu lộ trình
#đầu ra: m lộ trình tối ưu cho m xe, tổng thời gian hoàn thành tiếp quỹ ngắn nhất.
def toiuu(route,n):
    #route= route not depot
    #n=diem ngat
    route01=[]
    route02=[]
    total_time=100
    thoigianhoanthanh=0
    for i in range (1,len(route)):
        route.insert(i,n)
        route_all,route1,route2=split(route,n)
        bestroute1,kcmin1=insertion(route1)
        bestroute2,kcmin2=insertion(route2)
        time1=totaltime(Vdc,kcmin1)
        time2=totaltime(Vdc,kcmin2)
        if(time1<time2):
            thoigianhoanthanh=time2
        else:
            thoigianhoanthanh=time1
        if(thoigianhoanthanh<total_time):
            total_time=thoigianhoanthanh
            route01=bestroute1
            route02=bestroute2
        route.remove(n)
    lotrinh=[]
    lotrinh.append(route01)
    lotrinh.append(route02)
    
    return lotrinh,total_time

def traodoi(route_1,route_2,total_time):
    r_new_1=[]
    r_new_2=[]
    route01=[]
    route02=[]
    if(len(route_1)>len(route_2)):
        route01=route_1
        route02=route_2
    else:
        route01=route_2
        route02=route_1
    total_time_new=total_time
    for i in range (len(route01)):
        x=len(route01)-i
        for j in range (1,x):
            moi=[]
            r2=[]
            r2=copy.copy(route02)
            moi=route01[i:i+j]
            r2.extend(moi)
            r1=copy.copy(route01)
            del r1[i:i+j]

            best_route_new1,kc_new_1=insertion(r1)
            best_route_new2,kc_new_2=insertion(r2)
        
            time_new_1=totaltime(Vdc,kc_new_1)
            time_new_2=totaltime(Vdc,kc_new_2)
            if(time_new_1<time_new_2):
                thoigianhoanthanh_new=time_new_2
            else:
                thoigianhoanthanh_new=time_new_1
            if(thoigianhoanthanh_new<total_time_new):
                total_time_new=thoigianhoanthanh_new
                r_new_1=best_route_new1
                r_new_2=best_route_new2
            
    lotrinh_new=[]
    lotrinh_new.append(r_new_1)
    lotrinh_new.append(r_new_2)
    return lotrinh_new,total_time_new

route_not_depot,random_route=create_random_route(20)
lotrinh,total_time=toiuu(route_not_depot,21)
print('random route: ',random_route)
print('lo trinh ngan nhat la: ',lotrinh)
print('vay thoi gian hoan thanh ngan nhat la:',total_time)
lotrinh_new,total_time_new=traodoi(lotrinh[0],lotrinh[1],total_time)
if(total_time_new<total_time):
    print("lo trinh new:",lotrinh_new)
    print('time moi:',total_time_new)
else:
    print("lo trinh giu nguyen",lotrinh)
    print('gia tri tim ban dau la nho nhat',total_time)


