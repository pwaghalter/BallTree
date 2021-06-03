import pytest
import random as r     
from BallTree import BallTree
from BallTree import FakeBallTree

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
  
pytest.main(["-v", "-s", "test_BallTree.py"])
