#!/usr/bin/env python
# coding: utf-8

# In[10]:


n=int(input('Enter no of nodes'))
for i in range(n):
    print(chr(65+i),end=',')
print(" are the nodes")


# In[11]:


d={}
for i in range(n):
    print("Enter the child nodes of"+chr(i+65))
    c=list(map(str,input(":").split()))
    if c[0]!='0':
        d[chr(i+65)]=c
    else:
        d[chr(65+i)]=[]
print(d)


# In[12]:


source=input("Enter source node:")
key=input("Enter destination node:")
open_list=[source]
closed_list=[]
print("Open List--Closed List")
print(''.join(open_list),''.join(closed_list),sep='--')
while open_list:
    source=open_list.pop(0)
    closed_list.append(source)
    if source==key:
        print('Required node is found',key)
        break
    for i in d[source]:
        if i not in closed_list and i not in open_list:
            open_list.append(i)
#PRINTING OPEN AND CLOSED LIST FOR EACH ITERATION.
    print(''.join(open_list),''.join(closed_list),sep='--')


# In[13]:


source=input("Enter source node:")
key=input("Enter destination node:")
open_list=[source]
closed_list=[]
print("Open List--Closed List")
print(''.join(open_list),''.join(closed_list),sep='--')
while open_list:
    source=open_list.pop()
    closed_list.append(source)
    if source==key:
        print('Required node is found',key)
        break
    for i in d[source]:
        if i not in closed_list and i not in open_list:
            open_list.append(i)
#PRINTING OPEN AND CLOSED LIST FOR EACH ITERATION.
    print(''.join(open_list),''.join(closed_list),sep='--')


# In[16]:


def UCS(graph, s, goal):
    # FRONTIER IS THE DICTIONARY THAT STORES THE PATH AND ITS COST.
    frontier = {s: 0}
    #EXPLORED IS A LIST THAT HAS ALL THE NODES THAT ARE ALREADY EXPLORED TO AVOID INFINITE LOOPING.
    explored = []
    while frontier:
        print(f"Open list: {frontier}")
        print(f"Closed list: {explored}")
        node = min(frontier, key=frontier.get)
        val = frontier[node]
        print(node, " : ", val)
        del frontier[node]
        if goal == node:
            return f"Goal reached with cost: {val}"
        explored.append(node)
        for neighbour, pathCost in graph[node].items():
            if neighbour not in explored or neighbour not in frontier:
                frontier.update({neighbour: val + pathCost})
            elif neighbour in frontier and pathCost > val:
                frontier.update({neighbour: val})
    return "Goal not found"

# A FUNCTION TO READ INPUT GRAPH
def create_graph():
    num_nodes = int(input("Enter number of nodes in graph: "))
    graph = {}
    nodes = input("Enter the nodes separated by space: ").split()
    for node in nodes:
        children = input(f"Enter the child nodes of {node}: ").split()
        graph[node] = {}
        for child in children:
            if child != '0':
                cost = int(input(f"Enter the cost from {node} to {child}: "))
                graph[node][child] = cost
    return graph

graph = create_graph()  # READING INPUT
s = input("Enter source node: ")
g = input("Enter goal node: ")
print(UCS(graph, s, g))  # FUNCTION CALL


# In[17]:


print(graph)


# In[35]:


def BFS(direction, graph, frontier, reached):
    if direction == 'F':  # FROM ONE SIDE(SAY FRONT F)
        d = 'c'
    elif direction == 'B':  # FROM ONE SIDE(SAY BACK B)
        d = 'p'
    node = frontier.pop(0)
    for child in graph[node][d]:
        if child not in reached:
            reached.append(child)
            frontier.append(child)
    return frontier, reached

def isIntersecting(reachedF, reachedB):
    intersecting = set(reachedF).intersection(set(reachedB))
    return list(intersecting)[0] if intersecting else -1

def BidirectionalSearch(graph, source, dest):
    frontierF = [source]
    frontierB = [dest]
    reachedF = [source]
    reachedB = [dest]
    while frontierF and frontierB:
        print("From front: ")
        print(f"\tFrontier: {frontierF}")
        print(f"\tReached: {reachedF}")
        print("From back: ")
        print(f"\tFrontier: {frontierB}")
        print(f"\tReached: {reachedB}")
        frontierF, reachedF = BFS('F', graph, frontierF, reachedF)
        frontierB, reachedB = BFS('B', graph, frontierB, reachedB)
        intersectingNode = isIntersecting(reachedF, reachedB)
        if intersectingNode != -1:
            print("From front: ")
            print(f"\tFrontier: {frontierF}")
            print(f"\tReached: {reachedF}")
            print("From back: ")
            print(f"\tFrontier: {frontierB}")
            print(f"\tReached: {reachedB}")
            print("Path found!")
            path = reachedF[:-1] + reachedB[::-1]
            return path
    print("No path found!")
    return []

def create_graph():
    graph = {}
    n = int(input("Enter number of nodes in graph: "))
    for i in range(n):
        node = chr(65 + i)
        graph[node] = {'c': [], 'p': []}
    for i in range(n):
        node = chr(65 + i)
        children = input(f"Enter the child nodes of {node} separated by space (enter '0' for no child): ").split()
        for child in children:
            if child != '0':
                if child not in graph:
                    print(f"Error: Node '{child}' doesn't exist in the graph.")
                    return None
                graph[node]['c'].append(child)
                graph[child]['p'].append(node)
    return graph


s = input("Enter source node: ")
g = input("Enter goal node: ")
graph = create_graph()
path = BidirectionalSearch(graph, s, g)
if len(path):
    print("Path:", path)


# In[39]:


import itertools

def number(word, digit_map):
    return int(''.join(str(digit_map[letter]) for letter in word))

def solve_cryptarithmetic(puzzle):
    # TRYING ALL POSSIBLE MOVES WITH CONSTRAINTS
    words = puzzle.split()
    print(words)
    unique_characters = set(''.join(words))
    print(unique_characters)
    if len(unique_characters) > 10:  # NO MORE THAN 10 UNIQUE CHARS (0-9)
        return "Invalid puzzle: More than 10 unique characters"
    leading_characters = set(word[0] for word in words)
    print(leading_characters)
    if len(leading_characters) > 2/+:
        return "Invalid puzzle: More than 2 words start with the same character"
    for digits in itertools.permutations(range(10), len(unique_characters)):
        digit_map = dict(zip(unique_characters, digits))
        if all(digit_map[word[0]] != 0 for word in leading_characters):
            if sum(number(word, digit_map) for word in words[:-1]) == number(words[-1], digit_map):
                return digit_map
    return "No solution found"

puzzle = input("Enter the cryptarithmetic puzzle (words separated by spaces): ")
#solution =
solve_cryptarithmetic(puzzle)
#print("Solution:", solution)


# In[43]:


import math

def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    if curDepth == targetDepth:
        return scores[nodeIndex]
    if maxTurn:
        return max(minimax(curDepth + 1, nodeIndex * 2, False, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1, False, scores, targetDepth))
    else:
        return min(minimax(curDepth + 1, nodeIndex * 2, True, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1, True, scores, targetDepth))

scores = list(map(int, input("Enter scores:").split()))
treeDepth = int(math.log(len(scores), 2))
print("The optimal value is:", minimax(0, 0, True, scores, treeDepth))


# In[44]:


MAX, MIN = 1000, -1000

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
        best = MIN
        for i in range(2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

values = list(map(int, input("Enter scores:").split()))
print("The optimal value is:", minimax(0, 0, True, values, MIN, MAX))


# In[47]:


print("If the puzzle is \n1 2 3\n4 6\n7 5 8\ninput shall be given as 1234_6758")
start = input("Enter puzzle start state:")
final = input("Enter puzzle expected end state:")

def allpossible(state):
    l = []
    index = state.index("_")
    if index in [0,1,3,4,6,7]: # MOVE RIGHT
        source = state[:]
        source[index], source[index+1] = source[index+1], source[index]
        l.append(''.join(source))
    if index in [1,2,4,5,7,8]: # MOVE LEFT
        source = state[:]
        source[index], source[index-1] = source[index-1], source[index]
        l.append(''.join(source))
    if index in [3,4,5,6,7,8]: # MOVE UP
        source = state[:]
        source[index], source[index-3] = source[index-3], source[index]
        l.append(''.join(source))
    if index in [0,1,2,3,4,5]: # MOVE DOWN
        source = state[:]
        source[index], source[index+3] = source[index+3], source[index]
        l.append(''.join(source))
    return l

def heuristics(state, final):
    state = list(state)
    final = list(final)
    c = 0
    for i in range(len(state)):
        if state[i] != final[i]:
            c += 1
    return c

def EightPuzzle(source, final):
    all_nodes = [source]
    closed_list = []
    d = {}
    cost = 0
    while source != final:
        closed_list.append(source)
        l = allpossible(list(source))
        heur = []
        for i in l:
            heur.append(heuristics(i, final)) 
        index = heur.index(min(heur))
        d[source] = l
        source = l[index]
        cost += 1
        closed_list.append(source)
    return d, closed_list

EightPuzzle(start, final) # FUNCUTION CALL


# In[48]:


def water_jug_dfs(jug1_cap, jug2_cap, target):
    explored = set()
    frontier = [(0, 0)]

    while frontier:
        current = frontier.pop()
        print(current)

        if current == target:
            return True
        explored.add(current)

       
        next_states = []

        # Pour water from Jug 2 to Jug 1
        pour_2_1 = (min(current[0] + current[1], jug1_cap),
                              max(0, current[1] - (jug1_cap - current[0])))
        if pour_2_1 not in explored:
            next_states.append(pour_2_1)

        # Pour water from Jug 1 to Jug 2
        pour_1_2 = (max(0, current[0] - (jug2_cap - current[1])),
                              min(current[0] + current[1], jug2_cap))
        if pour_1_2 not in explored:
            next_states.append(pour_jug1_2)

        # Empty Jug 2
        empty_jug2 = (current[0], 0)
        if empty_jug2 not in explored:
            next_states.append(empty_jug2)

        # Empty Jug 1
        empty_jug1 = (0, current[1])
        if empty_jug1 not in explored:
            next_states.append(empty_jug1)

        # Fill Jug 2
        fill_jug2 = (current[0], jug2_cap)
        if fill_jug2 not in explored:
            next_states.append(fill_jug2)

        # Fill Jug 1
        fill_jug1 = (jug1_cap, current[1])
        if fill_jug1 not in explored:
            next_states.append(fill_jug1)

        # Add next states to the frontier in reverse order
        frontier.extend(reversed(next_states))
        explored.update(next_states)

    return False

# Example usage:
jug1_cap = 5
jug2_cap = 3
target_state = (4, 0)

result = water_jug_dfs(jug1_cap, jug2_cap, target_state)

if result:
    print("Solution exists.")
else:
    print("No solution found.")


# In[ ]:




