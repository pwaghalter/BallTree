import math
import random as r
import statistics as s
import heapq
import pytest

class Node(object):
    """
    Node class for BallTree.
    
    Attributes:
        coords (tuple): coordinates in nth dimension
        data (string/int): data associated with the Node
        radius (float): distance from Node to furthest point in hypersphere
        left (Node): left child
        right (Node): right child
    """    
    
    def __init__(self, coords, data, radius=0, left=None, right=None):
        """
        Constructor for Node class.
        
        Parameters:
            coords (tuple): coordinates in nth dimension
            data (string/int): data associated with the Node
            radius (float): distance from node to furthest point in hypersphere
            left (Node): left child
            right (Node): right child
        """
        self.coords=coords
        self.data=data
        self.radius=radius
        self.left=left
        self.right=right
        
    def __str__(self):
        """
        str method for Node class.
        """
        
        return "("+str(self.coords)+", "+str(self.data)+")"
    
class BallTree(object):
    """
    BallTree class. 
    
    Attributes: 
        dim (int): highest dimension
        nItems (int): number of key data pairs
        root (Node): highest level Node
        
    Notes:
        Spatial data structure used to store multidimensional data.
        Duplicate keys are stored once and data is combined. One Node represents
        all duplicates, and stores all corresponding data.
        Curse of dimensionality causes structure to be inefficient in very high
        dimensions.
    """

    def __init__(self, dataList, dimension):
        """
        Constructor for BallTree class.
        
        Paraemters:
           dataList (2D list): rows are key data pairs, keys are tuples of 
                               coordinates, data is a list
           
           dimension (int): highest dimension
        """
        self.__dim = dimension
        
        #incremented in build
        self.__nItems = 0
        
        #use private build method to construct the BallTree
        self.__root = self.__build(dataList)

    def __len__(self):
        """ Return number of Nodes in BallTree. """
        return self.__nItems
    
    # Used in __build to calculate which dimension to divide key data pairs on.
    def findLargestSpread(self, dataPoints):  
        """
        Calculates dimension of largest spread.
        
        Parameters:
            dataPoints (2D list): list of lists of key data pairs
            
        """
        #initalize lists to keep track of largest/smallest coordinate
        #in each dimension
        large = [-(math.inf) for i in range(self.__dim)]
        small = [(math.inf) for i in range(self.__dim)]  
        
        #examine each point and find the largest and smallest coordinate 
        #in each dimension
        for pair in dataPoints:
            
            #check that each key has the correct dimension
            if len(pair[0])!=self.__dim:
                raise Exception("Key", pair[0], "has inconsistent dimensions.")
            
            #check whether this coordinate is the largest or smallest in its
            #dimension so far
            for i in range(self.__dim):
                if pair[0][i] > large[i]: large[i] = pair[0][i] 
                if pair[0][i] < small[i]: small[i] = pair[0][i]
        
        #calculate which dimension has the greatest difference between
        #its largest and smallest coordinate
        largestSpread = abs(large[0]-small[0])
        dimension = 0
        
        #find the dimension of greatest spread
        for i in range(1, self.__dim):
            if abs(large[i]-small[i])>largestSpread:
                dimension = i
                largestSpread = abs(large[i]-small[i])
        
        #return dimension of greatest spread
        return dimension 
    
    #calculates the median of N coordinates along the dimension specified by
    #client. 
    def medianOfN(self, dataPoints, dim, N):
        """
        Returns the median of N randomly selected coordinates.
        
        Parameters:
            dataPoints (2D list): list of lists of key data pairs
            dim (int): dimension of largest spread in dataPoints
            N (int): number of random points to consider median between
        """
        potentialMedians=[]
        potentialPoints={}
        
        #at least calculate median of 3 points
        if N<3: N=3
        
        #N must be odd for the median calculation to work
        if not N%2: N+=1
        
        #generate a list of N random data points along the dimension     
        for i in range(N):
            
            #randomly select data points
            temp = r.choice(dataPoints)
            
            #store the coordinates of the random key data pairs in a list
            potentialMedians+=[temp[0][dim]]
            
            #store the key data pairs in a dictionary, where the key
            #to access the key data pairs is the coordinate on the dimension
            #whose median is being calculated
            potentialPoints[temp[0][dim]]=tuple(temp)
            
            
        #calculate the median of the random points using statistics module
        median = s.median(potentialMedians)
        
        #return the key data pair corresponding to the median point
        return potentialPoints[median]
            
    # Returns the square of the distance. Expects each set of coordinates
    # to be passed in as a tuple. Both sets of coordinates must be the same size.
    def distance(self, coord1, coord2):
        """
        Distance method for BallTree class.
        
        Parameters:
            coord1 (tuple): first coordinates used in distance formula
            coord2 (tuple): second coordinates used in distance formula
        """
        distance=0
        
        #sum the distance between coordinates in each dimensiom
        for i in range(self.__dim):
            distance+=(coord1[i]-coord2[i])**2
            
        return distance
    
    # Private construction method invoked by constructor. Recursively defines
    # hyperspheres corresponding to each data point in data set passed in by 
    # client.
    def __build(self, dataPoints):
        """
        Constructs the BallTree.
        
        Parameters:
            dataPoints (2D list): list of lists of key data pairs
        """
        #increment count of nodes
        self.__nItems+=1
        
        #if there is only one point in the data set create a corresponding node
        if len(dataPoints)==1: 
            
            #if the point in the data set has the wrong dimension,
            #raise exception
            if len(dataPoints[0][0])!=self.__dim:
                raise Exception("Key has inconsistent dimensions.")
            
            return Node(dataPoints[0][0], dataPoints[0][1])
        
        #find dimension of greatest spread
        curDim = self.findLargestSpread(dataPoints)
        
        #calculate the approximate median
        #larger data sets will use a larger number of randomized points to
        #approximate the median so that it's a more accurate approximate
        medianKey, medianData = self.medianOfN(dataPoints, curDim, int(len(dataPoints)*.2))
                
        radius = 0
        left=[]
        right=[]
        
        #calculate the radius of the hypersphere and subdivide the data
        #left and right of the median
        for pair in dataPoints: 
            
            #find the distance between the farthest point at this level and
            #the median, which is the radius of the hypershere defined at this
            #level of the BallTree
            radius = max(radius, self.distance(pair[0], medianKey))
            
            #subdivide the data points into left and right sets of data, based
            #on whether coordinates are greater or less than the median point
            #on the dimension of greatest spread
            if pair[0][curDim]<=medianKey[curDim] and pair[0]!=medianKey:
                left+=[pair]
                
            elif pair[0][curDim]>medianKey[curDim]:
                right+=[pair]
            
            #deal with duplicate coordinates
            elif pair[0]==medianKey:
                
                #if any of the data associated with the key in the duplicate
                #has not been stored in the data associated with the key in 
                #the original, store it there
                for num in pair[1]:
                    if num not in medianData: 
                        medianData=medianData+[num]
                        
                        #keep the data sorted
                        medianData.sort()

        #create a node representing the median point
        n = Node(medianKey, medianData, radius)
        
        #recurse to define the median's left and right children
        if len(left): n.left = self.__build(left)
        if len(right): n.right = self.__build(right)
        
        return n
    
    # Returns data associated with key specfied by client if it can be found
    # in the BallTree
    def findExact(self, coords):
        """
        Wrapper method for __findExact in BallTree class.
        
        Parameters:
            coords (tuple): coordinates to find in the BallTree
        """
        if len(coords)!=self.__dim: return
        return self.__findExact(self.__root, coords)
    
    # Private method that recursively searches the BallTree for key
    # specified by client
    def __findExact(self, n, targetCoords):
        """
        Returns data of Node being searched for if Node can be found.
        Returns None if Node is not in the BallTree.
        
        Parameters:
            n (Node): Node being examined
            targetCoords (tuple): coordinates to find in the BallTree
        """
        
        #if we found the desired node, return its data
        if n.coords == targetCoords: return n.data
         
        else:
            #check if the point could fall within the left child's radius
            if n.left and \
               self.distance(targetCoords, n.left.coords)<=n.left.radius:
                
                #recurse into the left child
                left = self.__findExact(n.left, targetCoords)
                if left: return left
                
            #check if the point could fall within the right child's radius
            if n.right and \
               self.distance(targetCoords, n.right.coords)<=n.right.radius:
                
                #recurse into the right child
                right = self.__findExact(n.right, targetCoords)  
                if right: return right
    
    # Returns the k nearest neighbors key and data as a tuple. Client recieves
    # at maximum the number of nodes in the tree
    def kNearestNeighbor(self, k, targetCoords):
        """
        Wrapper method for __kNearestNeighbor.
        Returns list of k nearest neighbors to specified point.
        
        Parameters:
            k (int): number of neighbors to return; if k is larger than
                     number of nodes in the BallTree, then returns all nodes
                     of the BallTree
            targetCoords (tuple): coordinate whose neighbors to find
        """
        if k<=0: return None
        
        #use a max heap to keep track of k nearest neighbors
        maxHeap = [(-math.inf, None, None) for i in range(min(k, len(self)))]
        heapq.heapify(maxHeap)
        
        #use private method to calculate the k nearest neighbors
        maxHeap = self.__kNearestNeighbor(k, targetCoords, self.__root, maxHeap)
        
        ans=[]
        
        #iterate through the heap and only return the key data pairs, not
        #the distance to target coordinates, which was also stored in the heap
        while len(maxHeap)>0:
            ans+=[heapq.heappop(maxHeap)[1:]]
        return ans
    
    # Private method that recursively finds the k nearest neighbors to the
    # key specified by the client
    def __kNearestNeighbor(self, k, targetCoords, n, maxHeap):
        """
        Returns max heap of k nearest neighbors. 
        Indices of heap are tuples of distance to target coordinates, key, and
        data of nearest neighbors.
        
        Parameters:
            k (int): number of nearest neighbors to return
            targetCoords (tuple): coordinate whose neighbors to find
            n (Node): Node whose distance to the targetCoords is being examined
            maxHeap: max heap used to store k nearest neighbors
        """

        #keep track of the distance between current node and target coordinates
        newDist = self.distance(targetCoords, n.coords)
        
        #check whether the current node is closer to the target than the
        #farthest of the nearest neighbors
        if newDist<=abs(maxHeap[0][0]):
            heapq.heapreplace(maxHeap, (-newDist, n.coords, n.data))
        
        #check if circle defined by left child intersects circle defined by
        #target coordinates and the farthest of the nearest neighbors
        if n.left and self.circleIntersects(n.left.coords, n.left.radius, \
                                            targetCoords, abs(maxHeap[0][0])):
            
            #if it does, then descend into the left child because there is a 
            #possibility that it is a closer neighbor to target coordinates
            maxHeap = self.__kNearestNeighbor(k, targetCoords, n.left, maxHeap)
            
        #check if circle defined by right child intersects circle defined by
        #target coordinates and the farthest of the nearest neighbors     
        if n.right and self.circleIntersects(n.right.coords, n.right.radius, \
                                             targetCoords, abs(maxHeap[0][0])): 
            
            #if it does, then descend into the left child because there is a 
            #possibility that it is a closer neighbor to target coordinates            
            maxHeap = self.__kNearestNeighbor(k, targetCoords, n.right, maxHeap)
      
        return maxHeap
    
    # Checks whether two hyperspheres intersect in n-dimensional space
    def circleIntersects(self, circle1Coords, circle1R, circle2Coords, circle2R):
        """
        Returns boolean describing whether two circles intersect.
        
        Parameters:
            circle1Coords: center of first circle
            circle1R: radius of first circle
            circle2Coords: center of second circle
            circle2R: radius of second circle
        """
        return self.distance(circle1Coords, circle2Coords) <= \
               (circle1R+circle2R)**2     
    
class FakeBallTree(object):
    """
    FakeBallTree class. Used to test BallTree class.
    
    Attributes:
        self.__nItems (int): number of Nodes in FakeBallTree
        self.__dataPoints (2D list): 2D list of key data pairs
        self.__dimension (int): highest dimension of FakeBallTree
    """
    
    def __init__(self, dataPoints, dimension):
        """
        Constructor for FakeBallTree class.
        
        Parameters:
            dataPoints (2D list): rows are key data pairs, keys are tuples of 
                                  coordinates, data is a list
           
           dimension (int): highest dimension
        """
        
        #updated in build
        self.__nItems = 0
        
        #use private method to build the FakeBallTree
        self.__dataPoints = self.__build(dataPoints)
        
        self.__dimension = dimension
    
    # Private method to build the FakeBallTree
    def __build(self, dataPoints):
        """
        Constructs the FakeBallTree.
        
        Parameters:
            dataPoints (2D list): list of lists of key data pairs

        """
        ans=[]
        
        #keep of track of if key is a duplicate
        found=0
        
        #examine each key data pair
        for pair in dataPoints:
            for i in range(len(ans)):
                
                #if key is a duplicate, append its data to the original key,
                #if the data is not also a duplicate
                if ans[i][0]==pair[0]:
                    found=1
                    if pair[1] not in ans[i][1]: 
                        ans[i][1]+=pair[1]
                        
                        #keep the data sorted
                        ans[i][1].sort()

            #if the key was not a duplicate, add the key data pair to the 2D 
            #list being constructed  
            if found==0:
                ans+=[pair]
                
                #increment count of key data pairs in FakeBallTree
                self.__nItems+=1

            found=0
        return ans
            
    def __len__(self): 
        """Returns number of key data pairs in FakeBallTree"""
        return self.__nItems
    
    # Returns data associated with key specfied by client if it can be found
    # in the FakeBallTree    
    def findExact(self, coords):
        """
        Returns data of corresponding to key being searched for, if key can be
        foudn.
        Returns None if key is not in the BallTree.
        
        Parameters:
            coords (tuple): coordinates to find in the BallTree
        """
                
        #look at each key data pair in the FakeBallTree
        for pair in self.__dataPoints:
            
            #if target coordinates are found, return corresponding data
            if pair[0]==coords: return pair[1]
    
    # Returns the square of the distance. Expects each set of coordinates
    # to be passed in as a tuple. Both sets of coordinates must be the same size.    
    def distance(self, coord1, coord2):
        """
        Distance method for FakeBallTree class.
        
        Parameters:
            coord1 (tuple): first coordinates used in distance formula
            coord2 (tuple): second coordinates used in distance formula
        """
        distance=0
        
        #calculate distance using distance formula
        for i in range(self.__dimension):
            distance+=(coord1[i]-coord2[i])**2
        return distance
    
    # Returns the k nearest neighbors key and data as a tuple. Client recieves
    # at maximum the number of nodes in the tree    
    def kNearestNeighbor(self, k, targetCoords):
        """
        Wrapper method for __kNearestNeighbor.
        Returns list of k nearest neighbors to specified point.
        
        Parameters:
            k (int): number of neighbors to return; if k is larger than
                     number of key data pairs in the FakeBallTree, then returns
                     all key data pairs of the FakeBallTree
            targetCoords (tuple): coordinate whose neighbors to find
        """
        if k<=0: return None
        
        #If k is larger than number of key data pairs in the FakeBallTree, then
        #returns all key data pairs of the FakeBallTree
        k = min(k, self.__nItems)
        
        #create a heap to keep track of the k nearest neighbors
        maxHeap = [(-math.inf, None, None) for i in range(min(k, len(self)))]
        
        #examine each key data pair, and check if it is closer than
        #the farthest nearest neighbor found so far
        for pair in self.__dataPoints:
            newDist = self.distance(pair[0], targetCoords)
            
            #if key is closer to target that the farthest of the k nearest
            #neighbors, push it on the heap and pop off the fathest of the
            #nearest neighbors
            if newDist<abs(maxHeap[0][0]):
                heapq.heapreplace(maxHeap, (-newDist, pair[0], pair[1]))  
                
                #if at some point there were equidistant points to the 
                #target key, and now we have found points closer to the target,
                #remove all of the points that were that distance from the target
                if len(maxHeap)>k:
                    for element in maxHeap:
                        if abs(element[0])>abs(maxHeap[0][0]):
                            maxHeap.remove(element)
                            heapq.heapify(maxHeap)
                                               
                    
            #if two points are equidistant to the target, include both of them
            #in the answer heap since we don't know which the real BallTree
            #included
            elif newDist==abs(maxHeap[0][0]):
                heapq.heappush(maxHeap, (-newDist, pair[0], pair[1])) 
        
        #return only the key and data of the k nearest neighbors, not the
        #corresponding distance to the target coordinates, which is also
        #in the heap returned by the private method
        ans=[]
        while len(maxHeap)>0:
            ans+=[heapq.heappop(maxHeap)[1:]]
        return ans
    
############################# TEST CODE #######################################
              
#Utility function to generate 2D list of random key data pairs. Takes in a 
#size for the 2D list and dimension for keys
def generateDataPoints(size, n):
    ans = []
    for i in range(size):
        key = []
        
        #create an n-dimension key whose coordinates can be positive or
        #negative
        for j in range(n):
            key.append(r.randint(-9999, 9999))
        
        #add the key and random data to the list of key data pairs
        ans.append([tuple(key), [r.randint(0, 9999)]])
            
    return ans

#test that all key data pairs are actually inserted in the BallTree
def test_allInserted():
    for i in range(10):
        
        #instantiate a BallTree of random size
        size = r.randint(1, 4)
        dataPoints = (generateDataPoints(r.randint(1, 999), size))
        b = BallTree(dataPoints, size)
        
        #check that each key data pair can be found after building the BallTree
        for pair in dataPoints:
            assert pair[1][0] in b.findExact(pair[0])
            

#Test that duplicates are not inserted twice in the BallTree
def test_duplicates():
    
    #generate random list of key data pairs
    size = r.randint(1, 4)
    dataPoints = (generateDataPoints(r.randint(1, 999), size))
    
    #accounting for duplicates in the original list before doubling it,
    #remember the size ofthe list of key data pairs before it is doubled
    temp = set()
    
    #each key will only be added once to the set since it does not allow duplicates
    for pair in dataPoints:
        temp.add(pair[0])
    
    #duplicate all the keys data pairs in the list of key data pairs
    dataPoints+=dataPoints
    
    #instantiate a BallTree and FakeBallTree with the list of duplicates
    b = BallTree(dataPoints, size)  
    f = FakeBallTree(dataPoints, size)
    
    #check that the size of the BalLTree and FakeBallTree are both the size
    #of the list of key data pairs before it's duplicated
    assert len(b)==len(f)==len(temp)

#Test that when every single key in the list to build the BallTree with is
#the same, only one Node is created and all of the data is inserted
def test_allKeysSame():
    size = 3
    
    #create a list of lists of key data pairs that all have the same key,
    #but have different data
    dataPoints = [[(1,2,3,), [r.randint(0, 100)]] for i in range(100)]
    
    b = BallTree(dataPoints, size)
    
    #make sure only one Node is created
    assert len(b)==1
    
    #cast the answer to a set for O(1) lookup
    ans = set(b.findExact((1,2,3)))
    
    #check that each piece of data was inserted
    for pair in dataPoints:
        assert pair[1][0] in ans
    

#Test findExact() on a simple data set whose answers are known
def test_simpleFindExact():
    
    #instantiate a BallTree using a very simple list of key data pairs
    size=2
    dataPoints = [[(1,2), [3]], [(1,3),[4]], [(5,6),[7]]]
    b = BallTree(dataPoints, size)
    
    assert b.findExact((1,2,))==[3]
    assert b.findExact((1,3,))==[4]
    assert b.findExact((5,6,))==[7]
    
#Test that findExact() works on a 2D BallTree when the key is in the tree
def test_findExact2DKeyThere():
    #instantiate a 2D BallTree and FakeBallTree
    size = 2
    dataPoints = (generateDataPoints(r.randint(1, 999), size))
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    key = r.choice(dataPoints)
    
    #search for a key known to be in the BallTree
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that both answers are defined and the same
    assert fakeAns and realAns
    assert fakeAns==realAns
    assert len(fakeAns)==len(realAns)    
    assert len(f)==len(b)

#Test that findExact() works on a 3D BallTree when the key is in the tree
def test_findExact3DKeyThere():
    #instantiate a #D BallTree and FakeBallTree
    size = 3
    dataPoints = (generateDataPoints(r.randint(1, 999), size))
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    key = r.choice(dataPoints)
    
    #search for a key known to be in the BallTree
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that both answers are defined and the same
    assert fakeAns and realAns
    assert fakeAns==realAns
    assert len(fakeAns)==len(realAns)    
    assert len(f)==len(b)

#Test that findExact() works on a BallTree of high dimension when the key is in
#the tree
def test_findExactHighDKeyThere():
    
    #instantiate a BallTree and FakeBallTree of random high dimension
    size = r.randint(4, 10)
    dataPoints = (generateDataPoints(r.randint(1, 999), size))
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    key = r.choice(dataPoints)
    
    #search for a key known to be in the BallTree
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that both answers are defined and the same
    assert fakeAns and realAns
    assert fakeAns==realAns
    assert len(fakeAns)==len(realAns)    
    assert len(f)==len(b)

#Test that findExact() works on a 2D BallTree when the key is not in the tree
def test_findExact2DKeyNotThere():
    
    #instantiate a BallTree and FakeBallTree
    size = 2
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #generate a random key
    key = generateDataPoints(1, size)[0]
    
    #make sure that the key is not in the BallTree
    while key[0] in dataPoints:
        key = generateDataPoints(1, size)[0]
    
    #search for the key
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that the answers are the same and not defined, since the key
    #is not in the BallTree
    assert fakeAns==realAns
    assert not (fakeAns or realAns)
    assert len(f)==len(b)
     
#Test that findExact() works on a 3D BallTree when the key is not in the tree
def test_findExact3DKeyNotThere():
    
    #instantiate a BallTree and FakeBallTree
    size = 3
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #generate a random key
    key = generateDataPoints(1, size)[0]
    
    #make sure that the key is not in the BallTree
    while key[0] in dataPoints:
        key = generateDataPoints(1, size)[0]
    
    #search for the key
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that the answers are the same and not defined, since the key
    #is not in the BallTree
    assert fakeAns==realAns
    assert not (fakeAns or realAns)
    assert len(f)==len(b)
      
#Test that findExact() works on a high dimension BallTree when the key is not in
#the tree    
def test_findExactHighDKeyNotThere():

    #instantiate a BallTree and FakeBallTree
    size = r.randint(4, 10)
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #generate a random key
    key = generateDataPoints(1, size)[0]
    
    #make sure that the key is not in the BallTree
    while key[0] in dataPoints:
        key = generateDataPoints(1, size)[0]
    
    #search for the key
    fakeAns = f.findExact(key[0])
    realAns = b.findExact(key[0])
    
    #check that the answers are the same and not defined, since the key
    #is not in the BallTree
    assert fakeAns==realAns
    assert not (fakeAns or realAns)
    assert len(f)==len(b)
    
#Test running findExact() on the same target coordinates multiple times.
def test_multipleFind():
    
    #Instantiate BallTree and FakeBallTree
    size = r.randint(1, 10)
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #generate random key
    key = generateDataPoints(1, size)[0]
    
    #remember first answer found when searching for the key
    ans = f.findExact(key[0])
    
    for i in range(10):
        fakeAns = f.findExact(key[0])
        realAns = b.findExact(key[0])
        
        if fakeAns and realAns:
            #check not only that FakeBallTree and BallTree give the same
            #answer, but that answer should be the same each time
            assert len(fakeAns)==len(realAns)==len(ans)
            
        assert (fakeAns)==(realAns)==ans
        assert (fakeAns and realAns) or (not fakeAns and not realAns)
        assert len(f)==len(b)
        
#Torture test for findExact()
def test_tortureFindExact():
    for i in range(r.randint(1, 99)):
        #generate random dimension BallTree and FakeBallTree
        size = r.randint(1, 10)
        dataPoints = generateDataPoints(r.randint(1, 999), size)
        f = FakeBallTree(dataPoints, size)
        b = BallTree(dataPoints, size)
        
        #generate a random key
        key = generateDataPoints(1, size)[0]
        
        #search for the key
        fakeAns = f.findExact(key[0])
        realAns = b.findExact(key[0])
   
        if fakeAns and realAns:
            assert len(fakeAns)==len(realAns)
            
        assert fakeAns==realAns
        assert (fakeAns and realAns) or (not fakeAns and not realAns)
        assert len(f)==len(b)

#Test very simple nearest neighbor search whose answer is known
def test_simplekNearestNeighbors():
    
    #Instantiate a BallTree using a very simple list of key data pairs
    size = 2
    dataPoints = [[(1,2), [3]], [(1,3),[4]], [(5,6),[7]]]
    b = BallTree(dataPoints, size)
    k = 1
    
    #check that the nearest neighbor is what it's known to be
    assert b.kNearestNeighbor(k, (4,5,))==[((5,6),[7])]

#Test kNearestNeighbor() when there are equidistant points from the target
def test_kNearestNeighborEquidistant():
    
    #create a very simple BallTree whose points are equidistant from
    #a search point
    size=2
    dataPoints = [[(1,1,),[4]], [(3,3,),[5]], [(1,3),[7]]]
    
    b = BallTree(dataPoints, size)
    f = FakeBallTree(dataPoints, size)
    
    #the target point equidistant to all points in the BallTree
    key = (2,2,2,)
    k = 2
    
    #find the k nearest neighbors to the point known to be equidistant
    realAns = b.kNearestNeighbor(k, key)
    fakeAns = f.kNearestNeighbor(k, key)
        
    #if two points are equidistant, and they are the farthest of the nearest 
    #neighbors, then both are stored in the FakeBallTree answer, but only one
    #will be included in the real BallTree answer
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
    
    assert len(realAns)==k
    
#Test that all nodes are still in the BallTree after performing a nearest
#neighbor search
def test_stillThereKNearestNeighbor():
    for i in range(10):
        
        #Instantiate a BallTree and a FakeBallTree of random dimension
        size = r.randint(1,10)
        dataPoints = generateDataPoints(r.randint(1, 999), size)
        f = FakeBallTree(dataPoints, size)
        b = BallTree(dataPoints, size)
        key = generateDataPoints(1, size)[0]
        k = r.randint(1, 15)
        
        #Perform nearest neighbor search
        fakeAns = f.kNearestNeighbor(k, key[0]) 
        realAns = b.kNearestNeighbor(k, key[0])
        
        assert len(realAns)<=len(fakeAns)
        for data in realAns: 
            assert data in fakeAns
    
        #check that real and fake classes returned the right number of neighbors
        if k<len(b): assert len(realAns)==k
        else: assert len(realAns)==len(b)        
   
        
        #examine each key data pair that was inserted in the BallTree and make
        #sure that it is still in the BallTree after kNearestNeighbor() is run
        for point in dataPoints:
            assert b.findExact(point[0])

#Test that kNearestNeighbor() works for negative value of k
def test_kNearestNeighborsNegative():

    #Instantiate BallTree and FakeBallTree using list of random key data pairs
    size = r.randint(1,10)
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    assert len(f)==len(b)
    
    for i in range(10):
        
        #Perform nearest neighbor search on random negative key    
        key = generateDataPoints(1, size)[0]
        k = r.randint(-30, 0)
        fakeAns = f.kNearestNeighbor(k, key[0]) 
        realAns = b.kNearestNeighbor(k, key[0])
        
        #check that real and fake classes had the same answer, and that answer
        #is None
        assert fakeAns == realAns == None
 
    
#Test that kNearestNeighbor search works in 1D
def test_kNearestNeighbors1D():
    
    #Instantiate BallTree and FakeBallTree using list of random key data pairs
    size = 1
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #Perform nearest neighbor search on random key    
    key = generateDataPoints(1, size)[0]
    k = r.randint(1, 15)
    fakeAns = f.kNearestNeighbor(k, key[0]) 
    realAns = b.kNearestNeighbor(k, key[0])
    
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
    
    #check that real and fake classes returned the right number of neighbors
    if k<len(b): assert len(realAns)==k
    else: assert len(realAns)==len(b)        


#Test that kNearestNeighbor search works in 2D
def test_kNearestNeighbors2D():
    
    #Instantiate BallTree and FakeBallTree using list of random key data pairs
    size = 2
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #Perform nearest neighbor search on random key    
    key = generateDataPoints(1, size)[0]
    k = r.randint(1, 15)
    fakeAns = f.kNearestNeighbor(k, key[0]) 
    realAns = b.kNearestNeighbor(k, key[0])
    
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
    
    #check that real and fake classes returned the right number of neighbors
    if k<len(b): assert len(realAns)==k
    else: assert len(realAns)==len(b)        
 

#Test that kNearestNeighbor search works in 3D
def test_kNearestNeighbors3D():
    
    #Instantiate BallTree and FakeBallTree using list of random key data pairs
    size = 3
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #Perform nearest neighbor search on random key    
    key = generateDataPoints(1, size)[0]
    k = r.randint(1, 15)
    fakeAns = f.kNearestNeighbor(k, key[0]) 
    realAns = b.kNearestNeighbor(k, key[0])
    
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
    
    #check that real and fake classes returned the right number of neighbors
    if k<len(b): assert len(realAns)==k
    else: assert len(realAns)==len(b)        


#Test that kNearestNeighbor search works in high dimension
def test_kNearestNeighborsHighD():
    
    #Instantiate BallTree and FakeBallTree using list of random key data pairs
    size = r.randint(4,10)
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    #Perform nearest neighbor search on random key    
    key = generateDataPoints(1, size)[0]
    k = r.randint(1, 15)
    fakeAns = f.kNearestNeighbor(k, key[0]) 
    realAns = b.kNearestNeighbor(k, key[0])
    
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
    
    #check that real and fake classes returned the right number of neighbors
    if k<len(b): assert len(realAns)==k
    else: assert len(realAns)==len(b)        

        
#Torture test for kNearestNeighbor()
def test_torturekNearestNeighbor():
    for i in range(50):
        
        #Instantiate BallTree and FakeBallTree using list of random key data pairs
        size = r.randint(1,10)
        dataPoints = generateDataPoints(r.randint(1, 999), size)
        f = FakeBallTree(dataPoints, size)
        b = BallTree(dataPoints, size)
        
        #Perform nearest neighbor search on random key    
        key = generateDataPoints(1, size)[0]
        k = r.randint(1, 15)
        fakeAns = f.kNearestNeighbor(k, key[0]) 
        realAns = b.kNearestNeighbor(k, key[0])
        
        #check that real and fake classes had the same answer
        assert len(realAns)<=len(fakeAns)
        for data in realAns: 
            assert data in fakeAns
        
        #check that real and fake classes returned the right number of neighbors
        if k<len(b): assert len(realAns)==k
        else: assert len(realAns)==len(b)        

#Test that kNearestNeighbor() gives at most the number of nodes in the 
#BallTree
def test_kNearestTooMany():
    
    #Instantiate a BallTree and FakeBallTree
    size = r.randint(1,10)
    dataPoints = generateDataPoints(r.randint(1, 999), size)
    f = FakeBallTree(dataPoints, size)
    b = BallTree(dataPoints, size)
    
    key = generateDataPoints(1, size)[0]
    
    #make sure k is larger than the number of key data pairs in the BallTree
    k = len(b)+r.randint(1, 15)

    fakeAns = f.kNearestNeighbor(k, key[0]) 
    realAns = b.kNearestNeighbor(k, key[0])
        
    #check that the right number of neighbors were returned
    assert len(realAns)==len(b)
    assert len(f)==len(b)    
    
    #check that real and fake classes had the same answer
    assert len(realAns)<=len(fakeAns)
    for data in realAns: 
        assert data in fakeAns
  
pytest.main(["-v", "-s", "BallTree.py"])
