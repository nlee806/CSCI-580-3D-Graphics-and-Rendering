#!/usr/bin/env python
# coding: utf-8

# In[41]:


from PIL import Image
import numpy as np
import json
import math
im = Image.new('RGB', [256,256], 0x000000)
width, height = im.size
xres = width
yres = height

#1. Loading JSON Data
with open('./teapot.json') as json_file:
    data = json.load(json_file)
#myjson = urllib2.urlopen('https://bytes.usc.edu/cs580/s21_cgMlArVr3D/hw/HW2/data/teapot.json')
#json_file = myjson.read()
#data = json.loads(json_file)

#print(data)
#print(data['data'][0]['v0']['v'])
#x y z (coordinates) nx ny nz (normal) u v (texture coordinates)

#background
for y in range(height):
    for x in range(width):
        im.putpixel((x,y), (128,112,96))

#initialize z-buffer to positive infinity
zBuffer = []
for a in range(height):
    bList = []
    for b in range(width):
        bList.append(float('inf'))
    zBuffer.append(bList)
     
#2. scan convert a single triangle
#Inputs: v0, v1, v2 - each an (x,y) [ignore z, normal, uv]; c (an RGB value)
def clipValues(v):
    if v[0] < 0:
        newa = (0, v[1], v[2])
        v = newa
    if v[1] < 0:
        newb = (v[0], 0, v[2])
        v = newb
    if v[2] < 0:
        newc = (v[0], v[1], 0)
        v = newc
    if v[0] > 255:
        newd = (255, v[1], v[2])
        v = newd
    if v[1] > 255:
        newe = (v[0], 255, v[2])
        v = newe
    if v[2] > 255:
        newf = (v[0], v[1], 255)
        v = newf
    return v
def scanTriangle(v0, v1, v2, r, g, b):
    #clip v0,v1,v2 (x,y) values to the buffer ie. (xres,yres)
    vb0 = clipValues(v0)
    vb1 = clipValues(v1)
    vb2 = clipValues(v2)
    xValues = [vb0[0], vb1[0], vb2[0]] 
    yValues = [vb0[1], vb1[1], vb2[1]] 
    xmin = int(math.floor(min(xValues)))
    xmax = int(math.ceil(max(xValues)))
    ymin = int(math.floor(min(yValues)))
    ymax = int(math.ceil(max(yValues)))
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    def f01(x,y):
        return (y0-y1)*x + (x1-x0)*y + x0*y1-x1*y0
    def f12(x,y):
        return (y1-y2)*x + (x2-x1)*y + x1*y2-x2*y1
    def f20(x,y): 
        return (y2-y0)*x + (x0-x2)*y + x2*y0-x0*y2
    for y in range(ymin,ymax):
        for x in range(xmin,xmax): 
            alpha = 0
            beta = 0
            gamma = 0
            alphaCheck = f12(x0,y0)
            betaCheck = f20(x1,y1)
            gammaCheck = f01(x2,y2)
            if (alphaCheck != 0):
                alpha = f12(x,y) / f12(x0,y0)
            if (betaCheck != 0):
                beta =  f20(x,y) / f20(x1,y1)
            if (gammaCheck != 0):
                gamma = f01(x,y) / f01(x2,y2)
            if ((alpha >= 0) and (beta >= 0) and (gamma >= 0)):
                zAtPixel = alpha*z0 + beta*z1 + gamma*z2
                if zAtPixel < zBuffer[x][y]:
                    im.putpixel((x,y),(r,g,b))
                    zBuffer[x][y] = zAtPixel

def computeTriangleColor(normal):
    dotp = float(float(0.707)*normal[0] + float(0.5)*normal[1] + float(0.5)*normal[2])
    if (dotp < 0.0):
        dotp = -dotp
    elif (dotp > 1.0):
        dotp = 1.0
    # "tint" the gray [for no good reason!]
    triColorRed = float(0.95)*dotp
    triColorGreen = float(0.65)*dotp
    triColorBlue = float(0.88)*dotp
    return [(triColorRed),(triColorGreen),(triColorBlue)]
    
##################################################################

def dot(A, B):
    #no negative signs in front of dot products
    return abs(A[0]*B[0] + A[1]*B[1] + A[2]*B[2])
    
def lenp(toLen):
    return math.sqrt(toLen[0]*toLen[0] + toLen[1]*toLen[1] + toLen[2]*toLen[2])
    
def unitize(toUnitize):
    l = lenp(toUnitize)
    toUnit = []
    for g in range(len(toUnitize)):
        if toUnitize[g] == 0:
            #toUnitize[g] = 0
            toUnit.append(0)
        else:
            #toUnitize[g] = 1
            toUnit.append(1)
    #return toUnit
    unitAnswer = []
    if l != 0:
        unitAnswer = [toUnitize[0]/l, toUnitize[1]/l, toUnitize[2]/l]
    else: 
        unitAnswer = [toUnitize[0], toUnitize[1], toUnitize[2]]
    #print("unitAnswer: ", unitAnswer, toUnit)
    #return unitAnswer
    return toUnit
    
def cross(A, B):
    #no negative signs in front of dot products
    C = []
    C.append((A[1]*B[2] - A[2]*B[1]))
    C.append((A[2]*B[0] - A[0]*B[2]))
    C.append((A[0]*B[1] - A[1]*B[0]))
    return C
    
def createCamMatrix(camR, to): #camera location->look at ex. 0,0,6 to 0,0,0
    camN = []
    for i in range(3):
        camN.append(camR[i]-to[i]) #from-to, from needs to be the tip
    camN = unitize(camN)
    camV = []
    camV.append(0)
    camV.append(1)
    camV.append(0) #fake V, just to create U
    camU = cross(camV, camN)
    camU = unitize(camU)
    camV = cross(camN, camU) #real V
    #print("camU: ", camU)
    #print("camV: ", camV)
    #print("camN: ", camN)
    #create camMat given camR, camU, camV, camN
    camMat = [[camU[0], camU[1], camU[2], (camR[0]*camU[0]+camR[1]*camU[1]+camR[2]*camU[2])], [camV[0], camV[1], camV[2], (camR[0]*camV[0]+camR[1]*camV[1]+camR[2]*camV[2])], [camN[0], camN[1], camN[2], (camR[0]*camN[0]+camR[1]*camN[1]+camR[2]*camN[2])],[0,0,0,1]]
    #print("camMat: ", camMat)
    return camMat

#convert world space to camera space
u = [1,0,0]
v = [0,1,0]
n = [0,0,1]
r = [0,0,20]
#the same as createCamMat, Camera
uvnrCamMatrix = np.array([[u[0],u[1],u[2],dot(r,u)],[v[0],v[1],v[2],dot(r,v)],[n[0],n[1],n[2],dot(r,n)],[0,0,0,1]])
#check1 = np.matmul(np.array([0,0,20,1]),uvnrCamMatrix)
#print("Check 1: should be 0,0,0:", check1) #wrong
#check2 = np.matmul(np.array([0,0,10,1]),uvnrCamMatrix)
#print("Check 2: should be 0,0,10:", check2)
#check3 = np.matmul(np.array([0,0,0,1]),uvnrCamMatrix)
#print("Check 3: should be 0,0,20:", check3) 
check1a = createCamMatrix([0,0,20], [0,0,0]) #i think this is the correct one. C = cam matrix = check1a
#check1ab = np.matmul(np.array(check1a), [0,0,20,1])
print("Check1a: should be 0,0,0", check1a) #i think this is right
#check2a = createCamMatrix([0,0,10], [0,0,0])
#print("Check2a: should be 0,0,10", check2a) #i think this is right
#check3a = createCamMatrix([0,0,0], [0,0,0])
#print("Check3a: should be 0,0,20", check3a)
#check4 = createCamMatrix([0,0,6], [0,0,0]) #at origin probably the right one
#print("Check 4: ", check4)
#check5 = np.matmul(np.array([0,0,6,1]),uvnrCamMatrix) #at camera
#print("Check 5: ", check5)
def worldToCam(coordinate):
    coordinate1 = []
    for c in range(len(coordinate)):
        if c == 1:
            coordinate1.append(-coordinate[c])
        else:
            coordinate1.append(coordinate[c])
    coordinate1.append(1)
    #NO 
    #result1 = createCamMatrix(coordinate, [0,0,0])
    coordinate1 = np.array(coordinate1)
    result1 = np.matmul(check1a, coordinate1)
    #print("World To Cam: ", result1)
    return perspectiveProjection(result1)

#projection x,y,z,1 -> x,y,z,w, only for vertices not for normals
near = 1
far = 5#100
top = 4#10
bottom = -4#-10
right = 4#10
left = -4#-10
#perspective projection
perspProj = np.array([[((2*near)/(right-left)), 0, -((right+left)/(right-left)), 0], [0, ((2*near)/(top-bottom)), -((top+bottom)/(top-bottom)), 0], [0, 0, ((far+near)/(far-near)), ((2*far*near)/(far-near))], [0, 0, -1, 0]])
#check4list = []#probably the right one
#for q in check4:
#    check4list.append(q[3])
#print("check4list", check4list)
#check4tomatrix = np.array(check4)
#print("Check 4 to matrix: ", check4tomatrix)
#check6 = np.matmul(check4tomatrix, perspProj)
#print("Check 6 4-persp:", check6) #this must be wrong
#check7 = np.matmul(check5, perspProj)
#print("Check 7 5-persp: ", check7)
#check8 = np.matmul(np.array(check4list), perspProj) #probably the right one
#print("Check 8:", check8)
def perspectiveProjection(result1):
    #result1list = []
    #for r in result1:
    #    result1list.append(r[3])
    #print("Perspective Project Result1 List: ", result1list)
    #result2 = np.matmul(np.array(result1list), perspProj)
    result2 = np.matmul(result1, perspProj)
    return toNDC(result2)

#divide x,y,z by w -> NDC vertices
#ppList7 = []
#ppList8 = []
#for t in range(3):
#    if check8[3] != 0:
#        ppList7.append(check7[t]/check8[3])
#        ppList8.append(check8[t]/check8[3])
#ppList7.append(check7[3])
#ppList8.append(check8[3])
#print("ppList7", ppList7)
#print("ppList8", ppList8) #this is the one I guess? NDC
def toNDC(result2):
    result3 = []
    #result2[2] = -result2[2] #negate z
    for p in range(3):
        if result2[3] != 0:
            result3.append(result2[p])#/-result2[3])
        else:
            result3.append(result2[p])
    #print("NDC result3: ", result3)
    return rasterSpace(result3)

#raster space
#xnd = ppList8[0]
#ynd = ppList8[1]
#znd = ppList8[2]
print("Final pixel: ",finalPixel)
def rasterSpace(result3):
    xnd = result3[0] 
    ynd = result3[1]
    znd = result3[2]
    xzero = 0
    yzero = 0
    xw = (xnd+1)*((width-1)/2)+xzero
    yw = (ynd+1)*((height-1)/2)+yzero
    zw = znd
    finalPixel = (xw,yw,zw)
    #print("Final pixel: ",finalPixel)
    return finalPixel

#############################################

#Object -> World (already)
for q in range(len(data['data'])):
    firstCoordinateXYZ = (data['data'][q]['v0']['v'][0], data['data'][q]['v0']['v'][1], data['data'][q]['v0']['v'][2])
    secondCoordinateXYZ = (data['data'][q]['v1']['v'][0], data['data'][q]['v1']['v'][1], data['data'][q]['v1']['v'][2])
    thirdCoordinateXYZ = (data['data'][q]['v2']['v'][0], data['data'][q]['v2']['v'][1], data['data'][q]['v2']['v'][2])
    firstNormalXYZ = (data['data'][q]['v0']['n'][0], data['data'][q]['v0']['n'][1], data['data'][q]['v0']['n'][2])
    secondNormalXYZ = (data['data'][q]['v1']['n'][0], data['data'][q]['v1']['n'][1], data['data'][q]['v1']['n'][2])
    thirdNormalXYZ = (data['data'][q]['v2']['n'][0], data['data'][q]['v2']['n'][1], data['data'][q]['v2']['n'][2])
    worldSpace = [firstCoordinateXYZ[0], firstCoordinateXYZ[1], firstCoordinateXYZ[2], secondCoordinateXYZ[0], secondCoordinateXYZ[1], secondCoordinateXYZ[2], thirdCoordinateXYZ[0], thirdCoordinateXYZ[1], thirdCoordinateXYZ[2], 0,0,0]
   
    tricolor = computeTriangleColor(data['data'][q]['v0']['n'])
    triColorR = max(0, min(255, int(math.floor(tricolor[0] * 256.0))))
    triColorG = max(0, min(255, int(math.floor(tricolor[1] * 256.0))))
    triColorB = max(0, min(255, int(math.floor(tricolor[2] * 256.0))))
    
    newC0 = worldToCam(firstCoordinateXYZ)
    newC1 = worldToCam(secondCoordinateXYZ)
    newC2 = worldToCam(thirdCoordinateXYZ)
    scanTriangle(newC0, newC1, newC2, triColorR, triColorG, triColorB)
    #scanTriangle(firstCoordinateXYZ, secondCoordinateXYZ, thirdCoordinateXYZ, triColorR,triColorG,triColorB)


#scan convert all triangles using z buffer
print("Finished.")
im = im.save("output.ppm")


# 
