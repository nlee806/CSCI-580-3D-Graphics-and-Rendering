#!/usr/bin/env python
# coding: utf-8

# 

# In[110]:


from PIL import Image
import numpy as np
import json
import math
im = Image.new('RGB', [512,512], 0x000000)
width, height = im.size
xres = 512
yres = 512

#1. Loading JSON Data
with open('./scene.json') as scene_file: 
    data2 = json.load(scene_file)


#######################################################
    
with open('./teapot.json') as json_file:
    data = json.load(json_file)
#myjson = urllib2.urlopen('https://bytes.usc.edu/cs580/s21_cgMlArVr3D/hw/HW2/data/teapot.json')
#json_file = myjson.read()
#data = json.loads(json_file)

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
    clipLimit = 512#255
    if v[0] < 0:
        newa = (0, v[1], v[2])
        v = newa
    if v[1] < 0:
        newb = (v[0], 0, v[2])
        v = newb
    if v[2] < 0:
        newc = (v[0], v[1], 0)
        v = newc
    if v[0] > clipLimit:
        newd = (clipLimit, v[1], v[2])
        v = newd
    if v[1] > clipLimit:
        newe = (v[0], clipLimit, v[2])
        v = newe
    if v[2] > clipLimit:
        newf = (v[0], v[1], clipLimit)
        v = newf
    return v
def scanTriangle(v0, v1, v2, normalMatrix, Cs, Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection):
    normalVector = normalMatrix
    #rasterize using x,y verts, each pixel interpolate z and use for hiding
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
            alphaCheck = f12(x0,y0)
            betaCheck = f20(x1,y1)
            gammaCheck = f01(x2,y2)
            if (alphaCheck == 0 or betaCheck == 0 or gammaCheck == 0): continue
            alpha = f12(x,y) / f12(x0,y0)
            beta =  f20(x,y) / f20(x1,y1)
            gamma = f01(x,y) / f01(x2,y2)
            if ((alpha >= 0) and (beta >= 0) and (gamma >= 0)):
                zAtPixel = alpha*z0 + beta*z1 + gamma*z2
                if zAtPixel < zBuffer[x][y]:
                    #
                    #Change Gouraud/Phong
                    gouraudNotPhong = False
                    #
                    vertexMatrix = [vb0, vb1, vb2]
                    if gouraudNotPhong == True: #gouraud, each pixel interpolate rgb from vertex lighting cals
                        finalColor = gouraud(vertexMatrix, alpha, beta, gamma, Cs, normalVector, Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
                    else: #phong, interpolate incoming normals
                        finalColor = phong(vertexMatrix, alpha, beta, gamma, Cs, normalVector, Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
                    tricolor = finalColor
                    r = max(0, min(255, int(math.floor(tricolor[0] * 256.0))))
                    g = max(0, min(255, int(math.floor(tricolor[1] * 256.0))))
                    b = max(0, min(255, int(math.floor(tricolor[2] * 256.0))))
                    zAtPixel = alpha*z0 + beta*z1 + gamma*z2
                    im.putpixel((x,y),(r,g,b))
                    zBuffer[x][y] = zAtPixel

#Flat Shading
def computeTriangleColor(normal): #1 normal of x,y,z
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

def unitize(v):
    d = math.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return [v[0]/d,v[1]/d,v[2]/d]
    
def cross(A, B):
    #no negative signs in front of dot products
    C = []
    C.append((A[1]*B[2] - A[2]*B[1]))
    C.append((A[2]*B[0] - A[0]*B[2]))
    C.append((A[0]*B[1] - A[1]*B[0]))
    return C
    
def createCamMatrix(camR, to): #camera location->look at ex. 0,0,6 to 0,0,0
    camN = np.array([camR[0]-to[0], camR[1]-to[1], camR[2]-to[2]])
    camN = unitize(camN)

    camV = np.array([0, 1, 0])

    camU = np.cross(camV, camN)
    camU = unitize(camU)

    camV = np.cross(camN, camU)

    cam_matrix = np.array([[camU[0], camU[1], camU[2], (camR[0]*camU[0]+camR[1]*camU[1]+camR[2]*camU[2])],
                          [camV[0], camV[1], camV[2], (camR[0]*camV[0]+camR[1]*camV[1]+camR[2]*camV[2])],
                          [camN[0], camN[1], camN[2], (camR[0]*camN[0]+camR[1]*camN[1]+camR[2]*camN[2])],
                          [0, 0, 0, 1]])
    return cam_matrix

def worldToCam(coordinate, check1b):
    #coordinate1 = []
    #for c in range(len(coordinate)):
        #if c == 1: #flip z to be negative
            #coordinate1.append(coordinate[c])
        #else:
            #coordinate1.append(coordinate[c])
    #coordinate1.append(1)
    #result1 = createCamMatrix(coordinate, [0,0,0])
    #coordinate1 = np.array(coordinate1)
    result1 = np.matmul(check1b, coordinate)
    return perspectiveProjection(result1)

#projection x,y,z,1 -> x,y,z,w, only for vertices not for normals
cameraBounds = data2['scene']['camera']['bounds']
near = cameraBounds[0]#3#0.25#1
far =  cameraBounds[1]#10#0.5 #5#100
top =  cameraBounds[4]#1#0.25 #4#10
bottom =  cameraBounds[5]#-1#-0.5 #-4#-10
right =  cameraBounds[3]#1#0.1 #4#10
left =  cameraBounds[2]#-1#-0.5 #-4#-10
#perspective projection
perspProj = np.array([[2*near/(right-left), 0, -(right+left)/(right-left), 0],
[0, 2*near/(top-bottom), -(top+bottom)/(top-bottom), 0],
[0, 0, (near+far)/(far-near), 2*far*near/(far-near)],
[0, 0, -1, 0]])
def perspectiveProjection(result1):
    result2 = np.matmul(perspProj,result1)# wrong order before, should be perspProj * result1 not result1 * perspProj
    return toNDC(result2)

#divide x,y,z by w -> NDC vertices
def toNDC(result2):
    result3 = []
    #result2[2] = -result2[2] #flip z/negate z
    for p in range(3):
        if result2[3] != 0:
            result3.append(result2[p]/result2[3])#-result2[3]) #flip w/negate w
        else:
            result3.append(result2[p])
    return rasterSpace(result3)

#raster space
def rasterSpace(result3):
    xnd = result3[0] 
    ynd = result3[1]
    znd = result3[2]
    xzero = 0
    yzero = 0
    xw = (xnd+1)*((xres-1)/2)+xzero
    yw = (ynd+1)*((yres-1)/2)+yzero
    zw = (znd+1)*((512-1)/2)
    finalPixel = (xw,yw,zw)
    return finalPixel

#############################################

#RotateX  
def rotatex(R,V):
    #XXX flipped theta
    radian = -math.radians(R)
    vX = V[0]
    vY = V[1]
    vZ = V[2]
    rMatrix = np.array([[1,0,0,0],[0,np.cos(radian),-np.sin(radian),0],[0,np.sin(radian), np.cos(radian),0],[0,0,0,1]])
    #vMatrix = np.array([vX,vY,vZ,1])
    rResult = np.matmul(rMatrix, V)
    return rResult 

#RotateY
def rotatey(R,V):
    #XXX flipped theta
    radian = -math.radians(R)
    vX = V[0]
    vY = V[1]
    vZ = V[2]
    rMatrix = np.array([[np.cos(radian),0,np.sin(radian),0],[0,1,0,0],[-np.sin(radian), 0,np.cos(radian),0],[0,0,0,1]])
    #vMatrix = np.array([vX,vY,vZ,1])
    rResult = np.matmul(rMatrix, V)
    return rResult 

#RotateZ
def rotatez(R,V):
    #XXX flipped theta
    radian = -math.radians(R)
    vX = V[0]
    vY = V[1]
    vZ = V[2]
    rMatrix = np.array([[np.cos(radian),-np.sin(radian),0,0],[np.sin(radian),np.cos(radian),0,0],[0,0,1,0],[0,0,0,1]])
    #rMatrix = np.array([[math.cos(radian),-math.sin(radian),0,0],[math.sin(radian),math.cos(radian),0,0],[0,0,1,0],[0,0,0,1]])
    #vMatrix = np.array([vX,vY,vZ,1])
    rResult = np.matmul(rMatrix, V)
    return rResult 

#Scale
def scale(S, V):
    sX = S[0]
    sY = S[1]
    sZ = S[2]
    vX = V[0]
    vY = V[1]
    vZ = V[2]
    sMatrix = np.array([[sX,0,0,0],[0,sY,0,0],[0,0,sZ,0],[0,0,0,1]])
    #vMatrix = np.array([vX,vY,vZ,1])
    sResult = np.matmul(sMatrix, V)
    return sResult

#Translate
def translate(T, V):
    tX = T[0]
    tY = T[1]
    tZ = T[2]
    vX = V[0]
    vY = V[1]
    vZ = V[2]
    #XXX -tZ added
    tMatrix = np.array([[1,0,0,tX],[0,1,0,tY],[0,0,1,-tZ],[0,0,0,1]])
    #vMatrix = np.array([vX,vY,vZ,1])
    tResult = np.matmul(tMatrix, V)
    return tResult
    
def clamp(toClamp):    
    maxClamp = 1
    minClamp = 0
    return max(min(toClamp, maxClamp), minClamp)
    
def normalize(nVector): 
    nx = nVector[0]
    ny = nVector[1]
    nz = nVector[2]
    n = math.sqrt(nx*nx+ny*ny+nz*nz) 
    nVector[0] = nx/n 
    nVector[1] = ny/n 
    nVector[2] = nz/n
    #handling NaN when alpha beta gamma = 0 0 0
    for w in range(3):
        if math.isnan(float(nVector[w])):
            nVector[w] = 0
    #nVector[0] = (nVector[0]).fillna(0).astype(int)
    #nVector[1] = (nVector[1]).fillna(0).astype(int)
    #nVector[2] = (nVector[2]).fillna(0).astype(int)
    return nVector
    
#Smooth Shading
def ads(Cs, normalVector, Kd, Ka, Ks, Ie, Ia, n, lights, lightDirection):
    Cs = Cs
    N = normalize(np.array(normalVector)) #surface normal vector normalized
    L = normalize(lightDirection) #light ray direction vector normalized
    Ks = Ks
    Ie = directional1 #light intensity directional light
    Ia = ambient1
    E = [0,0,1]#eye ray direction vector normalized
    S = n #sharpening
    #testing cases coming up in shading
    testNL = dot(N,L)
    testNE = dot(N,E)
    #if (testNL>0 and testNE>0): #both positive
        #compute lighting model -continue
        #N = N
    if (testNL<0 and testNE<0): #both negative
        #flip normal, compute lighting model on backside of surface
        N = -N
    #elif ((testNL>0 and testNE<0)or(testNL<0 and testNE>0)): #both different signs
        #light, eye on opposite sides of surface so light contributes 0, skip it
        #return Cs
    NL = dot(N,L)
    R = np.array([2*NL*N[0]-L[0], 2*NL*N[1]-L[1], 2*NL*N[2]-L[2]]) #reflected normalized ray direction vector
    R = normalize(R)
    #dot(R,E) must be clamped to 0, [0,1] bounded range
    sumSpec = 0
    RE = dot(R,E)
    if (RE > 1): RE = 1;
    if (RE < 0): RE = 0;
    #for s in lights: #sum over all directional lights
    #if s['type'] == 'directional':
    sumSpec = sumSpec+Ks*Ie*(RE** S)
    sumDiffuse = 0
    #for d in lights: #sum over all directional lights
    #if d['type'] == 'directional':
    sumDiffuse = sumDiffuse+Kd*Ie*NL
    Ia = Ia #light intensity ambient light
    #adsColor = (Ks*sumSpec)+(Kd*sumDiffuse)+(Ka*Ia)
    adsColor = sumSpec+sumDiffuse+(Ka*Ia)
    Cs = Cs*adsColor
    return Cs

#Gouraud, ADS lighting calcs at a vert, interpolate rgbs at each pixel
def gouraud(vertexMatrix, alpha, beta, gamma, Cs, normalVector, Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection):
    gColor = []
    n0 = normalVector[0]
    n1 = normalVector[1]
    n2 = normalVector[2]
    # ads calculation for vertices with normals of 3 vertices
    adsResult1 = ads(Cs, [n0[0],n0[1],n0[2]], Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
    adsResult2 = ads(Cs, [n1[0],n1[1],n1[2]], Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
    adsResult3 = ads(Cs, [n2[0],n2[1],n2[2]], Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
    #interpolation with three adsResult
    adsResult = []
    adsResult.append((adsResult1[0]*alpha)+(adsResult2[0]*beta)+(adsResult3[0]*gamma)) #interpolate red color
    adsResult.append((adsResult1[1]*alpha)+(adsResult2[1]*beta)+(adsResult3[1]*gamma)) #intierpolate green color
    adsResult.append((adsResult1[2]*alpha)+(adsResult2[2]*beta)+(adsResult3[2]*gamma)) #interpolate blue color
    gColor = adsResult
    return gColor

#Phong, interpolate vert's normals, use resulting normal to calculate ADS at each pixel
def phong(vertexMatrix, alpha, beta, gamma, Cs, normalVector, Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection):
    pColor = []
    n0 = normalVector[0]
    n1 = normalVector[1]
    n2 = normalVector[2]
    N0 = (n0[0] * alpha) + (n1[0] * beta) + (n2[0] * gamma)
    N1 = (n0[1] * alpha) + (n1[1] * beta) + (n2[1] * gamma)
    N2 = (n0[2] * alpha) + (n1[2] * beta) + (n2[2] * gamma)
    adsResult = ads(Cs, [N0,N1,N2], Kd, Ka, Ks, directional1, ambient1, n, lights, lightDirection)
    pColor = adsResult
    return pColor

shapes = data2['scene']['shapes']
lights = data2['scene']['lights']
ambientColor = lights[0]['color']
Ia = np.array(data2['scene']['lights'][0]['intensity'])
directionalColor = lights[1]['color']
Ie = np.array(data2['scene']['lights'][1]['intensity'])
lightFrom = lights[1]['from']
lightFrom[0] = -lightFrom[0]
lightFrom[1] = -lightFrom[1]
lightFrom[2] = lightFrom[2]
lightTo = lights[1]['to']
cameraFrom = data2['scene']['camera']['from']
cameraFrom[0] = -cameraFrom[0]
cameraFrom[1] = -cameraFrom[1]
cameraFrom[2] = cameraFrom[2]
cameraTo = data2['scene']['camera']['to'] #lookAt
cameraRes = data2['scene']['camera']['resolution']
xres = cameraRes[0]
yres = cameraRes[1]

for q in shapes:#range(len(shapes)):
    Cs = q['material']['Cs']
    Ka = q['material']['Ka']
    Kd = q['material']['Kd']
    Ks = q['material']['Ks']
    n = q['material']['n']
    Ry = q['transforms'][0]['Ry']
    ScaleMatrix = q['transforms'][1]['S']
    T = q['transforms'][2]['T']
    #Object -> World (already)
    for q in range(len(data['data'])):
        firstCoordinateXYZ = (data['data'][q]['v0']['v'][0], data['data'][q]['v0']['v'][1], data['data'][q]['v0']['v'][2])
        secondCoordinateXYZ = (data['data'][q]['v1']['v'][0], data['data'][q]['v1']['v'][1], data['data'][q]['v1']['v'][2])
        thirdCoordinateXYZ = (data['data'][q]['v2']['v'][0], data['data'][q]['v2']['v'][1], data['data'][q]['v2']['v'][2])
        firstNormalXYZ = (data['data'][q]['v0']['n'][0], data['data'][q]['v0']['n'][1], data['data'][q]['v0']['n'][2])
        secondNormalXYZ = (data['data'][q]['v1']['n'][0], data['data'][q]['v1']['n'][1], data['data'][q]['v1']['n'][2])
        thirdNormalXYZ = (data['data'][q]['v2']['n'][0], data['data'][q]['v2']['n'][1], data['data'][q]['v2']['n'][2])
        
        firstCoordinateXYZ = np.array([firstCoordinateXYZ[0],firstCoordinateXYZ[1],firstCoordinateXYZ[2],1])
        secondCoordinateXYZ = np.array([secondCoordinateXYZ[0],secondCoordinateXYZ[1],secondCoordinateXYZ[2],1])
        thirdCoordinateXYZ = np.array([thirdCoordinateXYZ[0],thirdCoordinateXYZ[1],thirdCoordinateXYZ[2],1])       
        firstNormalXYZ = np.array([firstNormalXYZ[0],firstNormalXYZ[1],firstNormalXYZ[2],1])
        secondNormalXYZ = np.array([secondNormalXYZ[0],secondNormalXYZ[1],secondNormalXYZ[2],1])
        thirdNormalXYZ = np.array([thirdNormalXYZ[0],thirdNormalXYZ[1],thirdNormalXYZ[2],1])
        
        #teapot.json*rotate 
        firstCoordinateXYZ = rotatey(Ry,firstCoordinateXYZ)
        secondCoordinateXYZ = rotatey(Ry,secondCoordinateXYZ)
        thirdCoordinateXYZ = rotatey(Ry,thirdCoordinateXYZ)
        firstNormalXYZ = rotatey(Ry,firstNormalXYZ)
        secondNormalXYZ = rotatey(Ry,secondNormalXYZ)
        thirdNormalXYZ = rotatey(Ry,thirdNormalXYZ)
        #scale
        #scale(S,V) 
        #normals do inverse of S * vertices
        firstCoordinateXYZ = scale(ScaleMatrix,firstCoordinateXYZ)
        secondCoordinateXYZ = scale(ScaleMatrix,secondCoordinateXYZ)
        thirdCoordinateXYZ = scale(ScaleMatrix,thirdCoordinateXYZ)
            #Inverse Normal Matrix
        #invsX = S[0]
        #invsY = S[1]
        #invsZ = S[2]
        #invsMatrix = np.array([[invsX,0,0,0],[0,invsY,0,0],[0,0,invsZ,0],[0,0,0,1]])
        #invsMatrix = np.array([firstCoordinateXYZ, secondCoordinateXYZ, thirdCoordinateXYZ, [0,0,0,1]])
        #inverseS1 = np.linalg.pinv(invsMatrix) #[1/sx, 1/sy, 1/sz],n0
        #invTrans = inverseS1.transpose()
        #invTrans = np.transpose(inverseS1)
            #this is the same as Inverse Normal Matrix
        inverseS = np.array([1/ScaleMatrix[0],1/ScaleMatrix[1],1/ScaleMatrix[2]])
        firstNormalXYZ = scale(inverseS,firstNormalXYZ)
        secondNormalXYZ = scale(inverseS,secondNormalXYZ)
        thirdNormalXYZ = scale(inverseS,thirdNormalXYZ)
        #translate
        #translate(T,V)
        firstCoordinateXYZ = translate(T,firstCoordinateXYZ)
        secondCoordinateXYZ = translate(T,secondCoordinateXYZ)
        thirdCoordinateXYZ = translate(T,thirdCoordinateXYZ)
        
        #mult by cam matrix - Normals too
        #mult by NDC matrix 
        #divide x,y,z by w, xy for rasterizing, z for zbuffering
        check1b = createCamMatrix(cameraFrom, cameraTo)
        newC0 = worldToCam(firstCoordinateXYZ, check1b)
        newC1 = worldToCam(secondCoordinateXYZ, check1b)
        newC2 = worldToCam(thirdCoordinateXYZ, check1b)
        vertMatrix = np.array([newC0, newC1, newC2])#change the naming of this, this is vertex coords, this is not camera matrix
        #newN0 = worldToCam(firstNormalXYZ, check1b)
        #newN1 = worldToCam(secondNormalXYZ, check1b)
        #newN2 = worldToCam(thirdNormalXYZ, check1b)
        #newN0 = np.matmul(check1b,firstNormalXYZ)#mutiply normal only with camera matrix, what you did before transfered the normal all the way to rasterization
        #newN1 = np.matmul(check1b,secondNormalXYZ)
        #newN2 = np.matmul(check1b,thirdNormalXYZ)
        normalMatrix = np.array([firstNormalXYZ, secondNormalXYZ, thirdNormalXYZ])
        #multiply light by camera matrix
        ambient1 = np.array(ambientColor*Ia)
        directional1 = np.array(directionalColor*Ie)
        directionalLight = (np.array(lightTo)-np.array(lightFrom)) #[-10,-5,0,1]
        directionalLight = [directionalLight[0], directionalLight[1], directionalLight[2], 1] #-10,-5,0,1
        #lightDirection = np.matmul(check1b,directionalLight)
        #DC = np.matmul(directionalLight,vertMatrix)#move light to camera matrix by mutipling it with camera matrix, but your camera matrix is check1b, camMatrix is your verx coords
        #do ads calculations on verts (which are in cam space)
        scanTriangle(vertMatrix[0], vertMatrix[1], vertMatrix[2], normalMatrix, Cs, Kd, Ka, Ks, directional1, ambient1, n, lights, directionalLight)
        #scanTriangle(firstCoordinateXYZ, secondCoordinateXYZ, thirdCoordinateXYZ, triColorR,triColorG,triColorB)

#scan convert all triangles using z buffer
print("Finished.")
im = im.save("output.ppm")


# In[ ]:




