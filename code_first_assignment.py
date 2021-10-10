#!/usr/bin/env python
# coding: utf-8

# Regex Substitution

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
from sys import stdin
import re
n = input()
print(re.sub( r"(?<= )(&&|\|\|)(?= )", lambda x: "and" if x.group()=="&&" else "or", stdin.read()))


# Matrix Script
# 

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
import math
import os
import random
import re
import sys
import re

n, m = map(int, input().split())
a, b = [], ""
for _ in range(n):
    a.append(input())

for z in zip(*a):
    b += "".join(z)

print(re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", b))



first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)


# Validating Postal Codes
# 
# 

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
regex_integer_in_range = r"_________"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"_________"	# Do not delete 'r'.
import re
s=input()
print (bool(re.match(r'^[1-9][\d]{5}$',s) and len(re.findall(r'(\d)(?=\d\1)',s))<2 ))


# Validating Credit Card Numbers

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
import re
TESTER = re.compile(
    r"^"
    r"(?!.*(\d)(-?\1){3})"
    r"[456]"
    r"\d{3}"
    r"(?:-?\d{4}){3}"
    r"$")
for _ in range(int(input().strip())):
    print("Valid" if TESTER.search(input().strip()) else "Invalid")


# Validating UID

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
import re

for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


# Detect HTML Tags, Attributes and Attribute Values

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]
        
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# HTML Parser - Part 2

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
from html.parser import HTMLParser

class HT(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print ('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print (data)
        
    def handle_data(self, data):
        if data !='\n':
            print('>>> Data')
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = HT()
parser.feed(html)
parser.close()


# HTML Parser - Part 1
# 
# 

# In[ ]:


#VISTO LA SOLUZIONE NELLA SEZIONE DISCUSSION

from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print ('Start :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
    def handle_endtag(self, tag):
        print ('End   :',tag)
        
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))


# Hex Color Code

# In[ ]:


#VISTO LA SEZIONE DISCUSSION PER SOLUZIONE
import re
pattern=r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]'
for i in range(int(input())):
    for x in re.findall(pattern,input()):
        print(x)


# Validating and Parsing Email Addresses
# 
# 

# In[ ]:


#SOLUZIONE VISTA NELLA SEDE DISCUSSION
import re
n = int(input())
for i in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)


# Validating phone numbers

# In[ ]:


#VISTO SEZIONE DISCUSSION E ADATTATO SOLUZIONE
import re
n=int(input())
for i in range(n):
    if re.match(r'[789]\d{9}$',input()):   
        print ('YES' ) 
    else:  
        print ('NO' )


# Validating Roman Numerals

# In[ ]:


##VISTO SOLUZIONE IN SEZIONE DISCUSSION

thousand = 'M{0,3}'
hundred = '(C[MD]|D?C{0,3})'
ten = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
regex_pattern=thousand + hundred+ten+digit +'$'

import re
print(str(bool(re.match(regex_pattern, input()))))


# Group(), Groups() & Groupdict()
# 
# 

# In[ ]:


#VISTO SEZIONE DISCUSSION E ADATTATO SOLUZIONE
#cerco le sequenze di almeno due caratteri alfanumerici consecutivi=m
#se m!=0 stampo la prima occorrenza, altrimenti stampo -1
import re
m = re.search(r'([a-zA-Z0-9])\1+', input().strip())
print(m.group(1) if m else -1)


# Decorators 2 - Name Directory

# In[ ]:


#VISTO LA SEZIONE DISCUSSION
import operator

def person_lister(f):
    def inner(people):
        # complete the function
        return map(f, sorted(people, key=lambda x: int(x[2])))          
    return inner
    

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# XML2 - Find the Maximum Depth

# In[ ]:


#VISTO SEZIONE DISCUSSION E ADATTATO LA SOLUZIONE
import xml.etree.ElementTree as etree
#funzione ricorsiva
maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1 
         
    for child in elem:
        depth(child, level + 1 ) 
    # your code goes here

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


# Athlete Sort

# In[ ]:


import math
import os
import random
import re
import sys
#VISTO SEZIONE DISCUSSION E ADATTATO LA SOLUZIONE
n, m = map(int, input().split())
nums = [list(map(int, input().split())) for i in range(n)]
k = int(input())
#ordino secondo l'elemento k-esimo
nums.sort(key=lambda x: x[k])

for line in nums:
    print(*line, sep=' ')


# Time Delta
# 
# 

# In[ ]:


#VISTO SEZIONE DISCUSSION E ADATTATO LA SOLUZIONE
import math
import os
import random
import re
import sys

# Complete the time_delta function below.
from datetime import datetime 

def time_delta(t1, t2):
    t1new=datetime.strptime( t1,'%a %d %b %Y %H:%M:%S %z')
    t2new=datetime.strptime(t2,'%a %d %b %Y %H:%M:%S %z')
    print (int(abs((t1new-t2new).total_seconds())))
    

t = int(input())

for t_itr in range(t):
    t1 = input()
    t2 = str(input())
    delta = time_delta(t1, t2)

   


# Piling Up!
# 
# 

# In[ ]:


#VISTO SEZIONE DISCUSSION E RIADATTATO LA SOLUZIONE
for j in range(int(input())):
    input()
    lst = list(map(int,input().split()))
    l = len(lst)
    i = 0
    while i < l - 1 and lst[i] >= lst[i+1]:
        i += 1
    while i < l - 1 and lst[i] <= lst[i+1]:
        i += 1
    if i==l-1:
        print("Yes")
    else:
        print("No")


# Company Logo

# In[ ]:


#VISTO LA SEZIONE DISCUSSION E RIFORMULATO LA SOLUZIONE
if __name__ == '__main__':
    s=str(input())
    Dizionario={}
#Ordino il dizionario mentre lo creo
    for x in sorted(s):
        Dizionario[x]=Dizionario.get(x,0)+1   
#ordino il dizionario per valori & metto le chiavi in Dict_keys.
    Dict_keys=sorted(Dizionario, key=Dizionario.get, reverse=True)  

    for key in Dict_keys[:3]:
        print(key,Dizionario[key])


# Word Order

# In[ ]:


from collections import defaultdict
n=(int(input()))

 
d=defaultdict(int)

for i in range(n):
    key=input()
    d[key] +=1
print(len(d.keys()))
    
occorrenza=(d.values())
print(*occorrenza)


# Collections.deque()
# 
# 

# In[ ]:


from collections import deque
N=int(input())
d=deque()

for i in range(N):
    comando=list(map(str,input().split()))
    if comando[0]=="append":
        d.append(comando[1])
    elif comando[0]=='pop':
        d.pop()     
    elif comando[0]=='popleft':
        d.popleft()    
    elif comando[0]=='appendleft':
        d.appendleft(comando[1])
print(' '.join(d))


# DefaultDict Tutorial
# 
# 

# In[ ]:


from collections import defaultdict

n,m=map(int,input().split())

A=[]
B=[]
for i in range(n):
    A.append(str(input()))
for i in range(m):
    B.append(str(input()))
    
for j in range(m):
    apparizione=[]
    for i in range(n):
        if B[j]==A[i]:
            apparizione.append(i+1)
    if apparizione:
        print(*apparizione, sep=' ')
    else:
        print(-1)


# Insertion Sort - Part 2
# 
# 

# In[ ]:



import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    # Write your code here
    for i in range(n):
        new=[]
        part1=arr[0:i]
        part2=arr[i+1:n]
        new=part1+part2
        j=0
        cont=0
        while j<i:
            if arr[i]>new[j]:
                cont=cont+1
            j+=1
            
        if cont>0:
            if cont==1:
                new.insert(cont,arr[i])
            else:
                new.insert(cont,arr[i])
        else:
            new.insert(0,arr[i])
        arr=new
        if i!=0:
            print(*new)   
            
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


# Insertion Sort - Part 1

# In[ ]:



import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
# 


def insertionSort1(n, arr):#doppio if per cambiare posizione iniziale
    current=arr[n-1]
    for i in range(1,n+1):
        if arr[n-i-1]>current:
            if i==n:
                arr[0]=current
            else:
                arr[n-i]=arr[n-i-1]
            print(*arr, sep = " ")
        if arr[n-i-1]<current:
            arr[n-i]=current
            print(*arr, sep = " ")
            break
            
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Recursive Digit Sum
# 
# 

# In[ ]:



import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
###RECURSIVE SOLUTION GIVES RUNNING TIME ERROR
'''
def superDigit(n, k):
    if (int(n)<10 and k<2):
        return (n)
    
    else:
        new=n*k
        new=str(new)
        if len(new)==1:
            return(new)
        somma=0
        for i in range(len(new)):
            somma+=int(new[i])
        
        somma=str(somma)
        return superDigit(somma,1)
'''
#The Digit Sum of a Number to Base 10 is Equivalent to Its Remainder Upon Division by 9
def superDigit(n, k):
    x = ((int(n) * k) % 9)
    if x!=0:
        return(x)
    else:
        return(9)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Viral Advertising
# 
# 

# In[ ]:



import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    # Write your code here
    shared=5
    liked=2
    ris=2
    for i in range(1,n):
        shared= math.floor(shared/2)*3
        liked= math.floor(shared/2)
        ris+=liked
    return(ris)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# Number Line Jumps

# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    
    if( (x1>x2 and v1>v2)or(x2>x1 and v2>v1) ):
        return('NO')
    for i in range(10000):
        if (x1+v1*i==x2+v2*i):
            return('YES')
    else:
        return('NO')


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# Birthday Cake Candles
# 
# 

# In[ ]:



import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    massimo=max(candles)
    ris=0
    for i in range(len(candles)):
        if candles[i]==massimo:
            ris+=1
    return ris

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# Re.start() & Re.end()

# In[ ]:


import re
S=str(input())
k=str(input())

risultati=[]
for i in range(len(S)):
    new=S[i:]
    
    if len(k)<=len(new):    
        ris=re.search(k,new)
        if ris:
            if( (ris.start()+i,ris.end()-1+i) not in risultati ):
                risultati.append( (ris.start()+i,ris.end()-1+i) )

if len(risultati)>0:
    for i in range(len(risultati)):
        print(risultati[i])
else:
        print('(-1, -1)')    


# Re.findall() & Re.finditer()
# 
# 

# In[ ]:


#S=str(input())

import re
consonanti = '[qwrtypsdfghjklzxcvbnm]'

#    This is a pattern for "positive lookbehind". For example, (?<=B)A means that PyCharm will search for A, but only if there is B before it
#{n,}    
#n is a non negative integer. Matches at least n times.

#For example, o{2,} does not match the o in Bob and matches all the o's in "foooood."
#flag re.I==re.Ignorcase e non dice di prendere sia minuscole che maiuscole
a = re.findall('(?<=' + consonanti +')([aeiou]{2,})' + consonanti, input(), re.I)

#case 2 non va bene
if len(a)>0:
    for i in range(len(a)):
       print(a[i])
else:
    print(-1)
    


# Re.split()
# 
# 

# In[ ]:


regex_pattern = r"[.,]"	# Do not delete 'r'.
'''
import re
try:    
    [print(i) for i in re.split('[.,]', input()) if i]
except EOFError as e:
    print('ciao')
 '''   

import re
print("\n".join(re.split(regex_pattern, input())))


# Detect Floating Point Number

# In[ ]:


T=int(input())
for i in range(T):
    x=input()
   
    try:
        if float(x):
            print('True')
        if float(x)==0:
            print('False')
            
    except Exception as e:
        print('False')


# Standardize Mobile Number Using Decorators

# In[ ]:


def wrapper(f):
    def fun(l):
        for i in range(len(l)):
            if (l[i][0]=='+' or (int(l[i][0])==0 and len(l[i])==11)):
                l[i]=l[i][1:]
                if len(l[i])==12:
                    l[i]=l[i][2:]
            elif (len(l[i])==12):
                l[i]=l[i][2:]

        l.sort()

        for i in range(0,len(l)):

            if (int(l[i][0])==9 and (len(l[i])<=11)):
                print('+91'+" "+str(l[i][0:5])+" "+str(l[i][5:10]))

            elif (int(l[i][0])==9 and (len(l[i])==12)):
                print('+'+str(l[i][0:2])+" "+str(l[i][2:7])+" "+str(l[i][7:12]) )

            elif(len(l[i])>12):
                print('+'+str(l[i][0:]))

            elif (int(l[i][0])==0):
                print('+91'+" "+str(l[i][1:6])+" "+str(l[i][6:11]) )

            else:         
                print('+91'+" "+str(l[i][0:5])+" "+str(l[i][5:10]) )   
    
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 


# XML 1 - Find the Score

# In[ ]:


import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    score=0
    #itero il nodo e prendo gli items: ne considero il numero con len e aggiorno lo score
    for child in node.iter():
        score+=(len(child.items()))
    return score


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# Map and Lambda Function
# 
# 

# In[ ]:


cube = lambda x:x**3 # complete the lambda function 

def fibonacci(n):
    lista=[0,1]
    for i in range(2,n):
        lista.append(lista[i-2]+lista[i-1])
    return lista[0:n]
    # return a list of fibonacci numbers


# ginortS
# 
# 

# In[ ]:


s=str(input())

lettersl=''
lettersu=''
numbersodd=''
numberseven=''
for i in range(len(s)):
    if s[i].islower():
        lettersl=lettersl+s[i]
    elif s[i].isupper():
        lettersu=lettersu+s[i]
    elif s[i].isnumeric():
        if int(s[i])%2==0:
            numbersodd=numbersodd+s[i]
        else:
            numberseven=numberseven+s[i]
    

print("".join(sorted(lettersl))+"".join(sorted(lettersu))+"".join(sorted(numberseven))+"".join(sorted(numbersodd)))


# Zipped!

# In[ ]:


#N=students, X=subjects

N,X=map(int,input().split())
subject=[]
for i in range(X):
    subject.append(list(map(float,input().split())))
#unzippo le liste
for i in zip(*subject):
    print(sum(i)/X)
    


# Exceptions
# 
# 

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
t=int(input())

for i in range (t):
    #provo a svolgere l'operazione
    try:
        a,b=map(int,input().split())
        print (a//b)
    #se non funziona stampo l'errore considerando la Exception: o ZeroDivisionError o ValueError
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


# Calendar Module

# In[ ]:


import calendar
mese,giorno,anno=map(int,input().split())

wday = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
print(wday[calendar.weekday(anno, mese, giorno)])


# Collections.OrderedDict()
# 
# 

# In[ ]:


from collections import OrderedDict
n=int(input())
spesa=OrderedDict()

for i in range(n):
    item=input().split()
    nome=''
    for i in range(len(item)-1):
        nome=nome+" "+item[i]
    #la prossima linea di codice serve a togliere lo spazio bianco inziale per il print
    nome=nome[1:]
    costo=int(item[-1])
    if nome not in spesa:
        spesa[nome] = costo
    else:
        spesa[nome] +=costo
    
[print(nome,spesa[nome]) for nome in spesa]
     


# Collections.namedtuple()
# 
# 

# In[ ]:


from collections import namedtuple
n=int(input())
columns=input().split()
#definisco la namedtuple schede=spreadsheets
schede=namedtuple('scheda',columns)
somma=0

for i in range(n):
    input1,input2,input3,input4=input().split()
    scheda=schede(input1,input2,input3,input4)
    somma= somma+ int(scheda.MARKS)

print('{:.2f}'.format(somma/n))
    


# collections.Counter()

# In[ ]:


import collections

x = int(input())
shoe_sizes = collections.Counter(map(int, input().split()))
N = int(input())

guadagno = 0

for i in range(N):
    numero, costo = map(int, input().split())
    if (shoe_sizes[numero]>0): 
        guadagno += costo
        shoe_sizes[numero] -= 1

print (guadagno)


# Linear Algebra

# In[ ]:


import numpy
numpy.set_printoptions(legacy='1.13')
n=int(input())
Matrix = numpy.array([input().split() for i in range(n)], float)
print(numpy.linalg.det(Matrix))


# Polynomials
# 
# 

# In[ ]:


import numpy
p=list(map(float,input().split()))
##x=map(float,input())
#result=numpy.polyval(p,x)
print(numpy.polyval(p,float(input())))


# Inner and Outer
# 
# 

# In[ ]:


import numpy
A = numpy.array([input().split()], int)
B = numpy.array([input().split()], int)
print( numpy.inner(A,B)[0][0])
print(numpy.outer(A,B))


# Eye and Identity

# In[ ]:


import numpy
numpy.set_printoptions(legacy='1.13')
n,m=map(int,input().split())
print(numpy.eye(n,m))


# Zeros and Ones
# 
# 

# In[ ]:


import numpy
dimensions = tuple(map(int, input().split()))
print (numpy.zeros(dimensions, dtype = numpy.int))
print (numpy.ones(dimensions, dtype = numpy.int))


# Concatenate

# In[ ]:


import numpy

N,M,P=map(int,input().split())
Matrix1 = numpy.array([input().split() for i in range(N)], int)
Matrix2 = numpy.array([input().split() for i in range(M)], int)

print(numpy.concatenate((Matrix1,Matrix2),axis=0))


# Transpose and Flatten

# In[ ]:


import numpy

N,M=map(int,input().split())
Matrix = numpy.array([input().split() for i in range(N)], int)
print (Matrix.transpose())
print (Matrix.flatten())


# Arrays

# In[ ]:



def arrays(arr):
    # complete this function
    # use numpy.array
    array=numpy.array(arr,float)
    return(numpy.flip(array))


# Dot and Cross

# In[ ]:


import numpy
matrix1=[]
matrix2=[]
n=int(input())

for i in range (n):
    matrix1.append(list(map(int,input().split())))
for i in range (n):
    matrix2.append(list(map(int,input().split())))    
A=numpy.array(matrix1)
B=numpy.array(matrix2)
print(numpy.matmul(A,B))


# Mean, Var, and Std
# 
# 

# In[ ]:


import numpy
matrix=[]
n,m=map(int,input().split())
for i in range (n):
    matrix.append(list(map(int,input().split())))
    
A=numpy.array(matrix)
print(numpy.mean(A,axis=1))    
print(numpy.var(A,axis=0))
err=numpy.std(A,axis=None)
print(round(err,11))


# Min and Max

# In[ ]:


import numpy
matrix=[]
n,m=map(int,input().split())
for i in range (n):
    matrix.append(list(map(int,input().split())))
    
A=numpy.array(matrix)
minimo=numpy.min(A, axis=1)
print(numpy.max(minimo))



# Sum and Prod

# In[ ]:


import numpy
matrix=[]
n,m=map(int,input().split())
for i in range (n):
    matrix.append(list(map(int,input().split())))
A=numpy.array(matrix)
product=numpy.sum(A,axis=0)

print(numpy.prod(product))


# Floor, Ceil and Rint

# In[ ]:


import numpy as np

np.set_printoptions(legacy='1.13')

A = list(map(float, input().split()))

print (np.floor(A))
print (np.ceil(A))
print (np.rint(A))


# Array Mathematics
# 
# 

# In[ ]:



import numpy as np
n, m = map(int, input().split())
a, b = (np.array([input().split() for i in range(n)], dtype=int) for j in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')


# Shape and Reshape

# In[ ]:


import numpy
array=list(map(int,input().split()))
print (numpy.reshape(array,(3,3)))


# Check Strict Superset

# In[ ]:


setA=set(map(int,input().split()))
n=int(input())
cont=0
for i in range(n):
    setB=set(map(int,input().split()))
    if setA.issuperset(setB):
        cont=cont+1
    else:
        cont=cont

if cont==n:
    print('True')
else:
    print('False')


# Check Subset
# 
# 

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
cases=int(input())
for i in range(cases):
    numA=int(input())
    setA=set(map(int,input().split()))
    numB=int(input())
    setB=set(map(int,input().split()))
    if setA.issubset(setB):
        print('True')
    else:
        print('False')
    


# The Captain's Room

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
k=int(input())
lista=list(map(int,input().split()))

setA=set();  
setB=set();  
for i in lista:
    if  i in setA:
        setB.add(i)
    else:
        setA.add(i)
        
ris=setA.difference(setB)
print (list(ris)[0])


# Set Mutations
# 
# 

# In[ ]:


n=int(input())
setA = set(map(int, input().split()))
N=int(input())
for i in range(N):
    comando=input().split()
    newset=set(map(int, input().split()))
    if comando[0]=='update':
        setA.update(newset)
    if comando[0]=='intersection_update':
        setA.intersection_update(newset)
    if comando[0]=='difference_update':
        setA.difference_update(newset)
    if comando[0]=='symmetric_difference_update':
        setA.symmetric_difference_update(newset)

print(sum(setA))


# Set .symmetric_difference() Operation

# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
N=int(input())
setn = set(map(int, input().split()))
M=int(input())
setm=set(map(int, input().split()))

unione=setn.symmetric_difference(setm)
print(len(unione))


# Set .difference() Operation

# In[ ]:


N=int(input())
setn = set(map(int, input().split()))
M=int(input())
setm=set(map(int, input().split()))

unione=setn.difference(setm)
print(len(unione))


# Set .intersection() Operation

# In[ ]:


N=int(input())
setn = set(map(int, input().split()))
M=int(input())
setm=set(map(int, input().split()))

unione=setn.intersection(setm)
print(len(unione))


# Set .union() Operation

# In[ ]:


N=int(input())
setn = set(map(int, input().split()))
M=int(input())
setm=set(map(int, input().split()))

unione=setn.union(setm)
print(len(unione))


# Set .discard(), .remove() & .pop()

# In[ ]:


n = int(input())
s = set(map(int, input().split()))
N=int(input())

for i in range(N):
    comando=input().split()
    if comando[0]=='pop':
        s.pop()
    if comando[0]=='remove':
        s.remove(int(comando[1]))
    if comando[0]=='discard':
        s.discard(int(comando[1]))

print(sum(s))
    


# Set .add()

# In[ ]:


N=int(input())

lista=[]
for i in range(N):
    lista.append(input())
    
sett=set(lista)

print(len(sett))


# No Idea!

# In[ ]:


n,m=input().split()
n=int(n)
m=int(m)

array=input().split()
array=[int(x) for x in array]

A=input().split()
A=[int(x) for x in A]
setA=set(A)

B=input().split()
B=[int(x) for x in B]
setB=set(B)

ris=0
for i in range (len (array)):
    if array[i] in setA:
        ris=ris+1
    elif array[i] in setB:
        ris=ris-1
    else:
        ris=ris

print(ris)


# Symmetric Difference
# 
# 

# In[ ]:


M=int(input())
setm=input().split()

N=int(input())
setn=input().split()

#trasform string into int
setm1=[int(x) for x in setm]
setn1=[int(x) for x in setn]
setM=set(setm1)
setN=set(setn1)

newset1=setN.difference(setM)
newset2=setM.difference(setN)
newset=newset1.union(newset2)

for i in range(len(newset)):
    print(min(newset))
    newset.discard(min(newset))
    


# Introduction to Sets

# In[ ]:


def average(array):
    # your code goes here
    new=set(array)
    lung=len(new)
    result=sum(new)/lung
    return result


# Merge the Tools!

# In[ ]:


def merge_the_tools(string, k):
    n=len(string)
    for i in range(0, n, k):
        u = ""
        for j in string[i : i + k]:
            if j not in u:
               u += j          
        print(u)


# The Minion Game

# In[ ]:



def minion_game(string):
    # your code goes here
    Vowels = 0
    Consonants = 0
    for i in range(len(string)):
        #if vocale(s[i]):
        if string[i] in 'AEIOU':
            Vowels += (len(string)-i)
        else:
           Consonants += (len(string)-i)

    if Vowels > Consonants:
        print ("Kevin", Vowels)
    elif Vowels< Consonants:
        print ("Stuart", Consonants)
    else:
        print ("Draw")
    


# Text Alignment

# In[ ]:


#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# sWAP cASE

# In[ ]:


def swap_case(s):
    l=list(s)
    new=[x.lower() if x.isupper() else x.capitalize() for x in l]
    return (''.join(new))


# Capitalize!

# In[ ]:


def solve(s):
    new=s.split()
    new2=[i.capitalize() for i in new]
    for i in range(len(new2)):
        s=s.replace(new[i],new2[i])
    return(s)   
    
   


# Alphabet Rangoli

# In[ ]:


import string

def print_rangoli(size):
    # your code goes here
    
    alpha = string.ascii_lowercase
    L = []
    for i in range(size):
        s = "-".join(alpha[i:size])
        L.append((s[::-1]+s[1:]).center(4*size-3, "-"))
    #upper    
    for i in range(size-1):
        print(L[size-1-i])
    #central
    print (L[0])
    #inferior
    for i in range(1,size):
        print(L[i])  


# String Formatting

# In[ ]:


def print_formatted(number):
    # your code goes here

    w=len(bin(number)[2:])
    for i in range(1,number+1):
        decimal=str(i).rjust(w,' ')
        octal=oct(i)[2:].rjust(w,' ')
        hexadecimal=hex(i)[2:].upper().rjust(w,' ')
        binary=bin(i)[2:].rjust(w,' ')
        print(decimal,octal,hexadecimal,binary)


# Designer Door Mat

# In[ ]:


n,m=input().split()
n=int(n)
m=int(m)

#superior + center
for i in range(int(n/2+1)):
    if i==(n/2-0.5):
        print ('WELCOME'.center(m,'-'))
    else :
        print (('.|.'*(2*i+1)).center(m,'-'))
    
#inferior
for i in range(int(n/2),0,-1):
    print (('.|.'*(2*i-1)).center(m,'-'))


# Text Wrap

# In[ ]:



def wrap(string, max_width):
    newstring=''
    for i in range (0,len(string),max_width):
        for j in range(max_width):
            if i+j<len(string):
                newstring=newstring+string[i+j]
        print(newstring)
        newstring=''
    return ''


# String Validators

# In[ ]:


if __name__ == '__main__':
    s = input()
    
segnale1=False;
segnale2=False;
segnale3=False;
segnale4=False;
segnale5=False;
    
for i in range(len(s)):
    if (s[i].isalnum()):
        segnale1=True;
    if (s[i].isalpha()):
        segnale2=True;
    if (s[i].isdigit()):
        segnale3=True;
    if (s[i].islower()):
        segnale4=True;
    if (s[i].isupper()):
        segnale5=True;
    
print(segnale1)
print(segnale2) 
print(segnale3) 
print(segnale4) 
print(segnale5) 


# Find a string

# In[ ]:


def count_substring(string, sub_string):
    cont=0
    segno=0
    for i in range(len(string)):
        if string[i]==sub_string[0]:
            for j in range(len(sub_string)):
                if i+j<len(string):
                    if string[i+j]==sub_string[j]:
                        segno= segno+1
            if segno==len(sub_string):
                cont=cont+1
        segno=0
    return cont


# Mutations

# In[ ]:


def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    newstring=''.join(l)
    return newstring


# What's Your Name?

# In[ ]:



def print_full_name(first, last):
    # Write your code here
   return print( "Hello "+first+" " +last +"! You just delved into python." )


# String Split and Join

# In[ ]:


def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line 

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# Lists

# In[ ]:


if __name__ == '__main__':
    N = int(input())
    
lista=[]    
for i in range(N):
    comando=input().strip().split(" ")
    funzione=comando[0]
    argomenti=comando[1:]
    #eseguo le richieste 
    if funzione=="insert":
        lista.insert(int(argomenti[0]),int(argomenti[1]))
    elif funzione=="print":
        print(lista)
    elif funzione=="remove":
        lista.remove(int(argomenti[0]))
    elif funzione=="append":
        lista.append(int(argomenti[0]))
    elif funzione=="sort":
        lista.sort()
    elif funzione=="pop":
        lista.pop()
    else:
        lista.reverse()


# Tuples

# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())

lista=[int(x) for x in integer_list]
tupla=tuple(lista)
print(hash(tupla))


# Finding the percentage
# 
# 

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
somma=student_marks[query_name][0]+student_marks[query_name][1]+student_marks[query_name][2]
somma = "{:.2f}".format(somma/3)
print(somma)


# Nested Lists

# In[ ]:


if __name__ == '__main__':
    lista=[]
    n=int(input())
    for _ in range(n):
        name = input()
        score = float(input())
        lista.append([name,score])
        
#find runnerup grades
new=[x[1] for x in lista[0:n][:]]
minimo=min(new)
new=[x for x in new if x!=minimo]
runnerup=min(new)

#print runnerup_names
runnerup_names=[x[0] for x in lista if x[1]==runnerup]
runnerup_names.sort()
for i in range(len(runnerup_names)):
    print(runnerup_names[i])


# Find the Runner-Up Score!

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
winner=max(arr)
new=[x for x in arr if x!=winner]
runnerup=max(new)
print(runnerup)


# List Comprehensions

# In[ ]:


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

primo=[i for i in range(x+1)]
secondo=[i for i in range(y+1)]
terzo=[i for i in range(z+1)]
result= [[i,j,k] for i in primo for j in secondo for k in terzo if i+j+k!=n]
print(result)


# Print Function

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    print(*range(1, n+1), sep='')


# Write a function

# In[ ]:


def is_leap(year):
    leap = False
    if year%4==0:
        leap=True;
        if year%100==0:
            leap=False;
            if year%400==0:
                leap=True;
    
    # Write your logic here
    
    return leap


# Loops

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    for i in range (n):
        print (i*i)
        i+=1

