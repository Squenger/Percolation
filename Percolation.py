



from mpl_toolkits.mplot3d.art3d import Poly3DCollection as poly
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import pygame


#Possible directions in Z^2
dir = [(1, 0), (0, -1), (-1, 0), (0, 1)]

#Initial grid construction
def grid(state, n):
  for i in range(n):
    state.append([])
    for j in range(n):
      state[i].append([0, 0, 0, 0, 0])
"""
A node contains information about its 4 neighboring edges (0 if open, 1 if closed) and another
value that will be used later for a graph traversal
"""

#Deletes an edge for a node and its homologue
def delete(state, x, y, direction):
  x2, y2 = dir[direction][0] + x, dir[direction][1] + y
  state[x][y][direction] = 1
  state[x2][y2][(2 + direction) % 4] = 1

#Percolation execution
def percolator(state, n, p):
  for i in range(n):
    for j in range(n):
      for k in range(1, 3): #We traverse the graph by treating only the right and left edges for each node
        if random.random() < p:
          delete(state, i, j, k)

#Program that returns a percolated graph
def generate(n,p):
  g=[]
  grid(g,n)
  percolator(g,n,p)
  return g


##CONNECTED COMPONENT ANALYSIS
#Returns the list of open neighboring edges
def neighbors(state, n, x, y):
  neighbor_list = []
  for direction in range(4):
    x2, y2 = dir[direction][0] + x, dir[direction][1] + y
    if (x2 >= 0 and y2 >= 0 and x2 < n and y2 < n):
      if state[x][y][direction] == 1:
        neighbor_list.append((x2, y2))
  return neighbor_list

#Starting from a node and a number, returns the size of the associated connected component and marks all nodes that are part of it using a depth-first traversal
def marking(state, n, x, y, mark):
  counter = 0
  queue = []
  state[x][y][4] = mark
  queue.append((x,y))
  while queue != []:
    v = queue.pop(0)
    i, j = v
    for neighbor in neighbors(state, n, i, j):
      i2, j2 = neighbor
      if state[i2][j2][4] == 0:
        state[i2][j2][4] = mark
        counter += 1
        queue.append(neighbor)
  return counter

#Returns the size and number of the largest connected component
def largest_connected_component(state, n):
  mark = 1
  max_val = -1
  id_max = mark
  for i in range(1, n):
    for j in range(1, n):
      if state[i][j][4] == 0:
        mark += 1
        counter = marking(state, n, i, j, mark)
        if counter > max_val:
          max_val = counter
          id_max = mark
  return (id_max,max_val)


##GRAPHICAL DISPLAY OF THE GRAPH

def graphics(state, n, surface, size, id_max):
  ratio = size / n
  colors = {}
  for i in range(n):
    for j in range(n):
      for k in range(1, 3):
        mark = state[i][j][4]
        if mark in colors:
          mark_colors = colors[mark]
        else:
          if mark == id_max:
            mark_colors = (0, 0, 0)
          else:
            mark_colors = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
          colors[mark] = mark_colors
        i2, j2 = dir[k]
        i2, j2 = i + i2, j + j2
        
        if state[i][j][k] == 1:
          pygame.draw.line(surface, colors[mark], (i * ratio, j * ratio), (i2 * ratio, j2 * ratio))

def display(N=1000,p=0.49):
  state = []
  grid(state, N)
  percolator(state, N, p)
  id_max = largest_connected_component(state, N)[0]
  pygame.init()
  display_surface = pygame.display.set_mode((700,700))
  pygame.display.set_caption('Percolation dans Z^2')
  closed = False
  display_surface.fill((255, 255, 255))
  graphics(state, N, display_surface, 700, id_max)
  pygame.display.update()
  while not closed:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        closed = True
  pygame.quit()


#percolation_3D


def percolation_3D (n,p):
  #Create the Z^3 graph
  graphe = nx.grid_graph(dim=[n, n, n], periodic=True)

  #Remove each edge with a probability 1-p
  to_remove=[]
  for arete in graphe.edges():
    if random.random() > p:
      to_remove.append(arete)
  graphe.remove_edges_from(to_remove)

  #Find the connected components
  connected_components = list(nx.connected_components(graphe))

  #Find the largest connected component
  max_component = max(connected_components, key=len)

  #Display the graph edges with the connected components
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  colors = ['b', 'g', 'r', 'c', 'm', 'y']
  for arete in graphe.edges():
    u, v = arete
    if u in max_component and v in max_component:
      ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], c='k')
    else:
      color = random.choice(colors)
      ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], c=color)
  plt.show()


##Statistics
#Plots the average (on q repetition) size of the largest connected component as a function of p (probability)
def Monte_Carlo(n=500,q=10):
  L=[]
  P=[0.1,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.485,0.49,0.491,0.492,0.493,0.494,0.495,0.496,0.497,0.498,0.499,0.5,0.505,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.7,0.9,1]
  #U contains the values of p for which we are going to measure the size of the largest connexe component
  for p in P:
    S=0
    print(p)
    for i in range(q):
      g=[]
      grid(g,n)
      percolator(g,n,p)
      S+=largest_connected_component(g,n)[1]
    L.append(S/q)
  plt.plot(P,L)
  plt.grid()
  plt.show()

#The same thing in 3D
def MC3D(n,q):
  L=[];U=[0.1,0.2,0.21,0.22,0.23,0.24,0.242,0.244,0.246,0.247,0.248,0.249,0.25,0.255,0.26,0.28,0.29,0.3,0.32,0.35,0.36,0.38,0.4,0.5,0.6,0.8,0.9,1]
  for p in U:
    S=0
    print(p)
    for i in range(q):
      graphe = nx.grid_graph(dim=[n, n, n])
      to_remove=[]
      for arete in graphe.edges():
        if random.random() > p:
          to_remove.append(arete)
      graphe.remove_edges_from(to_remove)
      #Find the connected components
      connected_components = list(nx.connected_components(graphe))
      #Find the largest connected component
      max_component = max(connected_components, key=len)
      S+=len(max_component)
    L.append(S/q)
  plt.plot(U,L)
  plt.grid()
  plt.show()


#Plots the probability of having a largest connected component of size greater than n as a function of n
def size(n,p,q,l):
  L=[]
  for i in range(q):
    L.append(marking(generate(n,p,n,n,n/2,n/2,1)))
  X = [i for i in range(n/2)]; E=[]; P=[]
  for i in X:
    print (i)
    S=0
    for c in L:
      if c>=i:
        S+=1
    S=S/q
    P.append(S)
    E.append(np.exp(-(i/l)))
  plt.figure()
  plt.plot(X,P)
  plt.plot(X,E)
  plt.show()

#Plots theta(p) as a function of p for different values of n to see how credible the simulation is
def cred(q):
  N=[400,300,200,100,50,10];R=[]
  U=[0.1,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.485,0.49,0.491,0.492,0.493,0.494,0.495,0.496,0.497,0.498,0.499,0.5,0.505,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.7,0.9,1]
  for n_val in N:
    L=[]
    for d in U:
      S=0
      print(d,n_val)
      for i in range(q):
        g=[]
        grid(g,n_val)
        percolator(g,n_val,d)
        S+=largest_connected_component(g,n_val)[1]/n_val**2
      L.append(S)
    R.append(L)
  plt.figure()
  for i in range (len(R)):
    plt.plot(U,R[i],label=N[i])
  plt.legend(loc='upper left')
  plt.show()


##Complexity
def Measure_Time(Program, args):
  start = time()
  _ = Program(*args)
  end = time()
  return end - start

def average_time(Program, n, num_reps):
  total_time = 0
  for p in np.linspace(0, 1, num_reps) :
    total_time += Measure_Time(Program, (n,p))
  return total_time/num_reps

def Complexity(Program, min_val = 100, max_val = 2000, num_points = 20, num_reps = 2) :
  X = np.linspace(min_val, max_val, num_points)
  Y = []
  i = 0
  for x in X :
    Y.append((average_time(Program, int(x), num_reps))**1/2)
    i += 1
    print(str(100*i/len(X))+"%")
  plt.plot(X, Y)
  plt.grid(True)
  plt.show()
  return X, Y