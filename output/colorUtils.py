'''
Created on 20.08.2015

@author: mkamp
'''
import itertools
from decimal import Decimal 
import sys

class ColorUtil:
    def __init__(self):
        pass
    
    def generateDistinctColors(self, num):
        if num < 1:
            print "Error: Stop screwing around."
            sys.exit()
          
        def MidSort(lst):
            if len(lst) <= 1:
                return lst
            i = int(len(lst)/2)
            ret = [lst.pop(i)]
            left = MidSort(lst[0:i])
            right = MidSort(lst[i:])
            interleaved = [item for items in itertools.izip_longest(left, right) for item in items if item != None]
            ret.extend(interleaved)
            return ret
        
        # Build list of points on a line (0 to 255) to use as color 'ticks'
        max = 255 #max is < 255 to avoid too light colors.
        segs = int(num**(Decimal("1.0")/2))
        step = int(max/segs)
        l = [(i*step) for i in xrange(1,segs)]
        points = [0,max]
        points.extend(MidSort(l))
        
        # Not efficient!!! Iterate over higher valued 'ticks' first (the points
        #   at the front of the list) to vary all colors and not focus on one channel.
        colors = ["#%02X%02X%02X" % (points[0], points[0], points[0])]
        range = 0
        total = 1
        while total < num and range < len(points):
            range += 1
            for c0 in xrange(range):
                for c1 in xrange(range):
                    for c2 in xrange(range):
                        if total >= num:
                            break
                        if points[c0] + points[c1] + points[c2] > 630: #in order to avoid too light colors
                            continue            
                        c = "#%02X%02X%02X" % (points[c0], points[c1], points[c2])
                        if c not in colors:
                            colors.append(c)
                            total += 1
        return colors