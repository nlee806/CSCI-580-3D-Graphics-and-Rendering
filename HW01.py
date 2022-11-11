#!/usr/bin/env python
# coding: utf-8

# In[62]:


from PIL import Image
im = Image.new('RGB', [512,512], 0x000000)
#print(im.size)
width, height = im.size

f = open("rects", "r")
f2 = f.readlines()

images = []
suprememax = 0
minClip = 0
maxClipX = width
maxClipY = height

for q2 in f2:
    l2 = q2.split()
    if len(l2) == 7:
        rgbs = [int(l2[4]), int(l2[5]), int(l2[6]), suprememax]
        suprememax = max(rgbs)
#print(suprememax)

for q in f2:
    l = q.split() #xmin, ymin, xmax, ymax, r, g, b
    if len(l) == 7:
        for w in range(0,4):
            if int(l[w]) < minClip: #clip negative values
                l[w] = minClip
        l[0] = int(l[0])
        l[1] = int(l[1])
        l[2] = int(l[2])
        l[3] = int(l[3])
        l[4] = int(round((int(l[4])/int(suprememax))*255))
        l[5] = int(round((int(l[5])/int(suprememax))*255))
        l[6] = int(round((int(l[6])/int(suprememax))*255))     
        images.append(l)
f.close()
#print(images)

#background
for y in range(height):
    for x in range(width):
        im.putpixel((x,y), (128,112,96))

#render images
for i in images: #xmin, ymin, xmax, ymax, r, g, b
    xmin = i[0]
    ymin = i[1]
    xmax = i[2]
    ymax = i[3]
    r = i[4]
    g = i[5]
    b = i[6]
    #print(i)
    for y in range(ymax-ymin+1):
        for x in range(xmax-xmin+1):
            if(x+xmin<maxClipX and y+ymin<maxClipY):
                im.putpixel((x+xmin,y+ymin), (r,g,b))
        
#for y in range(32):
#    for x in range(32):
#        im.putpixel((x,y), (255, 0, 0))

#for y in range(32):
#    for x in range(32):
#        im.putpixel((x+64, y+64), (255,255,0))

im

im = im.save("output.ppm")


# In[ ]:




