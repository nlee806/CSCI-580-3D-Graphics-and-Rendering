#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image
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
    xValues = [vb0[0], vb1[0], vb2[0]] #(int(vb0[0]), int(vb1[0]), int(vb2[0]))
    yValues = [vb0[1], vb1[1], vb2[1]] #(int(vb0[1]), int(vb1[1]), int(vb2[1]))
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
            alpha = f12(x,y) / f12(x0,y0)
            beta =  f20(x,y) / f20(x1,y1)
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
    
for q in range(len(data['data'])):
    firstCoordinateXYZ = (data['data'][q]['v0']['v'][0], data['data'][q]['v0']['v'][1], data['data'][q]['v0']['v'][2])
    secondCoordinateXYZ = (data['data'][q]['v1']['v'][0], data['data'][q]['v1']['v'][1], data['data'][q]['v1']['v'][2])
    thirdCoordinateXYZ = (data['data'][q]['v2']['v'][0], data['data'][q]['v2']['v'][1], data['data'][q]['v2']['v'][2])
    tricolor = computeTriangleColor(data['data'][q]['v0']['n'])
    triColorR = max(0, min(255, int(math.floor(tricolor[0] * 256.0))))
    triColorG = max(0, min(255, int(math.floor(tricolor[1] * 256.0))))
    triColorB = max(0, min(255, int(math.floor(tricolor[2] * 256.0))))
    scanTriangle(firstCoordinateXYZ, secondCoordinateXYZ, thirdCoordinateXYZ, triColorR,triColorG,triColorB)
        
#scan convert all triangles using z buffer
print("Finished.")
im = im.save("output2.ppm")


# In[ ]:




