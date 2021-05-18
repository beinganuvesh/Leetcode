#!/usr/bin/env python
# coding: utf-8

# In[46]:


def NextPermutations(arr):
    if len(arr)==0:
        return [[]]
    
    smallerOutput=NextPermutations(arr[1:])
    
    ans=[]
    for i in smallerOutput:
        ans.append([arr[0]]+i)
        
    return ans
        


# In[47]:


a=[1,2,3]
NextPermutations(a)


# In[12]:


[3]+[]


# In[ ]:





# In[16]:


def BSearch(arr, target):
    low=0
    high=len(arr)
    while low<high:
        mid=(low+high)//2
        if target>arr[mid]:
            low=mid+1
        elif target<arr[mid]:
            high=mid
        else:
            return mid
        
    return -1


# In[21]:


arr=[1,4,7,11,15]
t=18
BSearch(arr, t)


# In[ ]:





# In[28]:


def numIslands(grid):
    m=len(grid)
    n=len(grid[0])
    visited=[[False for j in range(n)]for i in range(m)]
    ans=0
    for i in range(m):
        for j in range(n):
            if visited[i][j]==False and grid[i][j]=="1":
                ans+=1
                Helper(i, j, m, n, grid, visited)
    return ans
      
      
def Helper(i, j, m, n, grid, visited):
  #Base Case!
  if i<0 or j<0 or i>=m or j>=n or grid[i][j]=="0" or visited[i][j]==True:
    return 
  visited[i][j]=True
  #Recursion!
  Helper(i+1, j, m, n, grid, visited)
  Helper(i, j+1, m, n, grid, visited)
  Helper(i-1, j, m, n, grid, visited)
  Helper(i, j-1, m, n, grid, visited)
  


# In[29]:


grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
numIslands(grid)


# In[12]:


def Shift(n, arr):
     d={}
     for i in range(n):
          key=arr[i]
          if key in d:
               d[key].append(i)
          else:
               d[key]=[]
               d[key].append(i)
               
     s,e=d[1][0], d[1][-1]
     c=0
     for i in range(s,e+1):
          if arr[i]==0:
               c+=1    
     return c
          
          


# In[13]:


n=7
l=[0,0,1,0,1,0,1]
Shift(n, l)


# In[ ]:





# In[1]:


def Boring(n):
    dic={1:1, 2:3, 3:6, 4:10}
    r=n%10
    a1=(r-1)*10
    d=0
    while n!=0:
        d+=1
        n=n//10
    return a1+dic[d]
    


# In[5]:


Boring(1)


# In[ ]:





# In[85]:


import heapq
class MedianFinder:
    def __init__(self):
        self.left=[]
        self.right=[]
    
    def addNum(self, num):
        if len(self.left)==0:
            heapq.heappush(self.left, -num)
        elif num<=self.left[0]:
            heapq.heappush(self.left, -num)
        else:
            heapq.heappush(self.right, num)
            
        #Check if the elements in both are not more than 1!
        if len(self.left)-len(self.right)>1:
          heapq.heappush(self.right, -heapq.heappop(self.left))
        elif len(self.right)-len(self.left)>1:
          heapq.heappush(self.left, -heapq.heappop(self.right))
          
    def findMedian(self):
        if len(self.left)>len(self.right):
          return -(self.left[0])
        elif len(self.left)<len(self.right):
          return self.right[0]
        else:
          return (self.right[0]-self.left[0])/2
        


# In[90]:


obj=MedianFinder()
obj.addNum(1)
obj.addNum(2)
obj.addNum(3)
obj.addNum(4)


# In[91]:


obj.findMedian()


# In[102]:


def sortColors(nums):
    d={}
    for i in nums:
        d[i]=d.get(i,0)+1
          
    for i in range(len(nums)):
        if 0 in d and d[0]>0:
            nums[i]=0
            d[0]-=1
        elif 1 in d and d[1]>0:
            nums[i]=1
            d[1]-=1
        else:
            nums[i]=2
            d[2]-=1


# In[103]:


nums = [2,0,2,1,1,0]
sortColors(nums)
nums


# In[ ]:





# In[85]:


def exist(board, word):
    m=len(board)
    n=len(board[0])
      
    for i in range(m):
        for j in range(n):
            if board[i][j]==word[0]:
                isTrue=Helper(board, i, j, m, n, 0, word)
                if isTrue:
                    return True
            
    return False
       

def Helper(board, i, j, m, n, idx, word):
    if idx==len(word):
        return True
    if i<0 or j<0 or i>=m or j>=n or board[i][j]!=word[idx]:
        return False
    
    temp=board[i][j]
    board[i][j]=" "
    x=Helper(board, i+1, j, m, n, idx+1, word)
    y=Helper(board, i, j+1, m, n, idx+1, word)
    z=Helper(board, i-1, j, m, n, idx+1, word)
    t=Helper(board, i, j-1, m, n, idx+1, word)
    board[i][j]=temp
    
    if x or y or z or t:
        return True
    else:
        return False


# In[86]:


board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "SEE"
exist(board, word)


# In[ ]:





# In[62]:


def uniquePaths(m, n):
    dp=[[0 for j in range(n)]for i in range(m)]
      
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            if i==m-1 and j==n-1:
                dp[i][j]=0
            elif i==m-1:
                dp[i][j]=1
            elif j==n-1:
                dp[i][j]=1
            else:
                dp[i][j]=dp[i+1][j]+dp[i][j+1]
            
    return dp[0][0]
            


# In[63]:


uniquePaths(3, 2)


# In[ ]:





# In[47]:


def merge(ar):
    arr=sorted(ar, key=lambda x:x[0])
    res=[]
    res.append(arr[0])
    
    for i in range(1, len(arr)):
        previous=res.pop()
        current=arr[i]
        if (previous[1]>=current[0]) and previous[1]<=current[1]:
            res.append([previous[0], current[1]])
        elif (previous[1]>=current[0]) and previous[1]>current[1]:
            res.append([previous[0], previous[1]])
        else:
            res.append([previous[0], previous[1]])
            res.append([current[0], current[1]])
            
    return res
    


# In[48]:


arr = [[1,4], [2,3]]
merge(arr)


# In[ ]:





# In[7]:


def Knows(a, b, matrix):
    if matrix[a][b]==1:
        return True
    else:
        return False

def Celebrity(n, matrix):
    a=0
    b=n-1
    
    while a<=b:
        if Knows(a, b, matrix):
            a+=1
        else:
            b-=1
        
    for i in range(n):
        if matrix[a][i]==1 or matrix[i][a]==0:
            return -1
        
    return a


# In[8]:


n=3
MATRIX = [ [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0] ]
Celebrity(n, MATRIX)


# In[ ]:





# In[103]:


x=sorted(strs[0])
s=""
s.join(x)


# In[112]:


def Anagrams(strs):
    d={}
    for word in (strs):
        root=sorted(word)
        s=""
        key=s.join(root) 
        value = word
        if key in d:
            d[key].append(value)
        else:
            d[key]=[]
            d[key].append(value)
    return d.values()


# In[113]:


strs = ["eat","tea","tan","ate","nat","bat"]
Anagrams(strs)


# In[ ]:





# In[42]:


def Rotate(arr):
    arr[:] = arr[::-1]
    arr[:] = [[arr[j][i] for j in range(len(arr))]for i in range(len(arr[0]))]
    return arr


# In[43]:


matrix =[[1,2,3],[4,5,6],[7,8,9]]
Rotate(matrix)        


# In[ ]:





# In[41]:


def WordBreak(s, d):
    if len(s)==0:
        return True
    
    for i in range(1, len(s)+1):
        if (s[0:i] in d) and (WordBreak(s[i:], d)):
            return True
    return False


# In[43]:


s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
WordBreak(s, wordDict)


# In[ ]:





# In[45]:


def WordBreak_Memo(s, d, dic):
    if len(s)==0:
        return True
    
    for i in range(1, len(s)+1):
        if (s[0:i] in d):
            if s[i:] in dic:
                ans = dic[s[i:]]
            else:
                ans = WordBreak_Memo(s[i:], d, dic)
                dic[s[i:]]=ans
            if ans:
                return True
    return False


# In[46]:


s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
WordBreak_Memo(s, wordDict, {})


# In[ ]:





# In[39]:


def findMedianSortedArrays(nums1, nums2):
    i=0
    j=0
    res=[]
    while i<len(nums1) and j<len(nums2):
        if nums1[i]<nums2[j]:
            res.append(nums1[i])
            i+=1
        elif nums1[i]>nums2[j]:
            res.append(nums2[j])
            j+=1
        else:
            res.append(nums1[i])
            res.append(nums2[j])
            i+=1
            j+=1
            
    while i<len(nums1):
        res.append(nums1[i])
        i+=1
    while j<len(nums2):
        res.append(nums2[j])
        j+=1
    
    l=len(res)
    if l%2==0:
        return (res[(l//2)-1]+res[(l//2)])/2
    else:
        return res[l//2]
        


# In[40]:


ar1=[1,2]
ar2=[3,4]
findMedianSortedArrays(ar1, ar2)


# In[12]:


from queue import PriorityQueue
def kLargest(n, arr, k):
    q=PriorityQueue()
    #Putting the first K elements.
    for i in range(k):
        q.put(arr[i])
        
    for i in range(k, n):
        current=q.get()
        if current<arr[i]:
            q.put(arr[i])
        else:
            q.put(current)
    
    ans=[]
    while q.empty() is False:
        ans.append(q.get())
    return ans
        


# In[14]:


n=4
l=[3,2,5,6]
k=2
x=kLargest(n, l, k)
x


# In[35]:


def subsets(A):
    l=[]
    Help(sorted(A), l, 0, [])
    return l
        
def Help(arr, l, index, r):
    l.append(list(r))
    for i in range(index, len(arr)):
        r.append(arr[i])
        Help(arr, l, i+1, r)
        r.pop()


# In[36]:


subsets(n)


# In[37]:


def Helper(arr):
    #Base Case 
    if len(arr)==0:
        l=[]
        l.append([])
        return sorted(l)
        
    smallerOutput=Helper(arr[1:])
    ans=[]
    for i in smallerOutput:
        ans.append(i)
        ans.append([arr[0]]+i)
  
    return ans


# In[38]:


n=[12,13]
Helper(n)


# In[72]:


def combine(n, k):
    numbers=[int(i) for i in range(1, n+1)]
    res=[]
    Helper(numbers, k, res, [], 0)
    return res

def Helper(numbers, k, res, r, start):
    if len(r)==k:
        res.append(list(r))
        return 
    
    for i in range(start, len(numbers)):
        r.append(numbers[i])
        Helper(numbers, k, res, r, i+1)
        r.pop()


# In[73]:


combine(4, 2)


# In[52]:


def CombinationSum(candidates, target):
    candidate=sorted(candidates)
    res=[]
    Helper(candidate, target, res, [], 0)
    return res
    
def Helper(candidate, target, res, r, start):
    if target<0:
        return 
    if target==0:
        res.append(list(r))
        return 
    
    for i in range(start, len(candidate)):
        r.append(candidate[i])
        Helper(candidate, target-candidate[i], res, r, i)
        r.pop()


# In[53]:


candidates = [2,3,5] 
target = 8
CombinationSum(candidates, target)


# In[38]:


def threeSum(num):
    nums=sorted(num)
    length=len(nums)
    lst=[]
        
    for i in range(length-1):
        target=nums[i]
        #Check for duplicacy
        if i>0 and nums[i]==nums[i-1]:
            continue
        
        l=i+1
        r=length-1
        
        while l<r:
          if target+nums[l]+nums[r]==0:
            lst.append([target, nums[l], nums[r]])
            
            while l<length-1:
                if nums[l]==nums[l+1]:
                    l+=1
                else:
                    break
                    
            while r>0:
                if nums[r]==nums[r-1]:
                    r-=1
                else:
                    break
            
            l+=1
            r-=1    
            
          elif target+nums[l]+nums[r]<0:
            l+=1
          else:
            r-=1
            
    return lst


# In[39]:


num = [-1,0,1,2,-1,-4]
x=[0,0,0]
threeSum(x)


# In[20]:


import sys
def threeSumclosest(num, k):
    nums=sorted(num)
    length=len(nums)
    ans=0
    d=sys.maxsize
        
    for i in range(length-1):
        target=nums[i]
        
        l=i+1
        r=length-1
        
        while l<r:
          if target+nums[l]+nums[r]==k:
            return k
        
          elif target+nums[l]+nums[r]<k:
            diff=abs(target+nums[l]+nums[r]-k)
            if diff<d:
                ans=target+nums[l]+nums[r]
                d=diff
            l+=1
                
          else:
            diff=abs(target+nums[l]+nums[r]-k)
            if diff<d:
                ans=target+nums[l]+nums[r]
                d=diff
            r-=1
            
    return ans


# In[21]:


n = [-1,2,1,-4]
m=[0,1,2]
k = 3
threeSumclosest(m, k)


# In[21]:


from queue import PriorityQueue
q = PriorityQueue()


# In[6]:


import heapq


# In[12]:


arr=[('Anuvesh', 0), ('Name', 3), ('is', 1), ('my', 2)]
l=[(0 ,'Anuvesh'), (3, 'Name'), (1, 'is'), (2, 'my')]


# In[19]:


class Pair:
    def __init__(self, n, dist, psf):
        self.n=n
        self.dist=dist
        self.psf=psf
        
def Function(arr):
    pq=[]
    for i in range(len(arr)):
        heapq.heappush(pq, (arr[i][1],arr[i]))
    
    while len(pq)>0:
        current=heapq.heappop(pq)
        print(current)


# In[20]:


Function(arr)


# In[16]:


Function(l)

