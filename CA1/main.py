import operator
from queue import LifoQueue, PriorityQueue, Queue
from random import shuffle
from typing import List


def moveTuple(pos: tuple, d: tuple) -> tuple:
    return tuple(map(operator.add, pos, d))

def manhattanDistance(a: tuple, b: tuple) -> int:
    x1, y1 = a
    x2, y2 = b
    return abs(x2 - x1) + abs(y2 - y1)

class Map:
    def __init__(self, ambulance: tuple, hospitals: dict, patients: list, obstacles: list):
        self.ambulance = ambulance
        self.hospitals = dict(hospitals)
        self.obstacles = list(obstacles)
        self.patients = list(patients)

    def canMovePatient(self, pos: tuple, d: tuple) -> bool:
        newPos = moveTuple(pos, d)
        return newPos not in self.obstacles and newPos not in self.patients

    def canMoveAmbulance(self, d: tuple) -> bool:
        newAmbulancePos = moveTuple(self.ambulance, d)
        return newAmbulancePos not in self.obstacles and \
            (newAmbulancePos not in self.patients or self.canMovePatient(newAmbulancePos, d))

    def move(self, d: tuple) -> bool:
        """
        Move the ambulance if d is a valid move
        and returns True, returns False otherwise
        """
        if not self.canMoveAmbulance(d):
            return False

        self.ambulance = moveTuple(self.ambulance, d)
        try:
            patientIndex = self.patients.index(self.ambulance) # raises ValueError
            newPatientPos = moveTuple(self.patients[patientIndex], d)

            if newPatientPos in self.hospitals and self.hospitals[newPatientPos]:
                self.hospitals[newPatientPos] -= 1
                self.patients.pop(patientIndex)
            else:
                self.patients[patientIndex] = newPatientPos

        except ValueError:
            pass

        return True

    def findNearestHospitalDistance(self, patient: tuple) -> int:
        minDist = -1
        for hospital in self.hospitals:
            dist = manhattanDistance(patient, hospital)
            minDist = dist if dist < minDist or minDist == -1 else minDist

        return minDist

    def findNearestHospitalToPatientsDistance(self) -> int:
        res = 0
        for patient in self.patients:
            res += self.findNearestHospitalDistance(patient)

        return res

    @property
    def isGoal(self) -> bool:
        return len(self.patients) == 0

    @classmethod
    def buildMapFromMap(cls, map):
        return cls(
            ambulance=map.ambulance,
            hospitals=map.hospitals,
            patients=map.patients,
            obstacles=map.obstacles
        )

    @classmethod
    def buildMapFromFile(cls, inputFileName: str):
        ambulance = ()
        hospitals = {}
        obstacles = []
        patients = []

        with open(inputFileName, 'r') as fin:
            for row, line in enumerate(fin):
                for col, cell in enumerate(line):
                    curr = (row, col)

                    if cell == 'A':
                        ambulance = curr
                    elif cell.isdigit():
                        hospitals[curr] = int(cell)
                    elif cell == '#':
                        obstacles.append(curr)
                    elif cell == 'P':
                        patients.append(curr)

        return cls(
            ambulance=ambulance,
            hospitals=hospitals,
            patients=patients,
            obstacles=obstacles
        )

    def __str__(self):
        return str((
            self.ambulance,
            tuple(self.hospitals.items()),
            tuple(self.patients)
        ))

    def __lt__(self, other):
        return len(self.patients) < len(other.patients)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((
            self.ambulance,
            tuple(self.hospitals.items()),
            tuple(self.patients)
        ))

class SearchProblem:
    class ResultGenerator:
        KEY_DEPTH = "Depth"
        KET_SUCCEEDED = "Succeeded"
        KEY_TOTAL_STATES = "Total States"
        KEY_UNIQUE_STATES = "Unique States"

        @classmethod
        def failure(cls) -> dict:
            return {
                cls.KET_SUCCEEDED: False,
                cls.KEY_DEPTH: -1,
                cls.KEY_TOTAL_STATES: -1,
                cls.KEY_UNIQUE_STATES: -1,
            }

        @classmethod
        def success(cls, depth=0, totalStatesCount=0, uniqueStatesCount=0) -> dict:
            return {
                cls.KET_SUCCEEDED: True,
                cls.KEY_DEPTH: depth,
                cls.KEY_TOTAL_STATES: totalStatesCount,
                cls.KEY_UNIQUE_STATES: uniqueStatesCount,
            }

    def __init__(self, inputFileName: str):
        self.inputFileName = inputFileName
        self.POSSIBLE_MOVES = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]
        shuffle(self.POSSIBLE_MOVES)

    def getStartState(self) -> Map:
        return Map.buildMapFromFile(self.inputFileName)

    def getSuccessors(self, state: Map) -> List[Map]:
        res: List[Map] = []

        for d in self.POSSIBLE_MOVES:
            newState = Map.buildMapFromMap(state)
            valid = newState.move(d)
            if valid:
                res.append(newState)

        return res

    def bfs(self):
        startState: Map = self.getStartState()
        if startState.isGoal:
            return self.ResultGenerator.success()

        queue: Queue[Map] = Queue()
        queue.put((startState, 0))

        visited = set()

        totalStatesCount = 0
        while not queue.empty():
            totalStatesCount += 1

            currentState, depth = queue.get()
            visited.add(currentState)

            depth += 1
            for state in self.getSuccessors(currentState):
                if state not in visited:
                    if state.isGoal:
                        return self.ResultGenerator.success(
                            depth=depth,
                            totalStatesCount=totalStatesCount,
                            uniqueStatesCount=len(visited)
                        )

                    queue.put((state, depth))

        return self.ResultGenerator.failure()

    def dls(self, startState: Map, maxDepth: int):
        stack: LifoQueue[Map] = LifoQueue()
        stack.put((startState, 0))

        visited = set()
        visitDepth = {}

        totalStatesCount = 0
        while not stack.empty():
            totalStatesCount += 1

            currentState, depth = stack.get()
            visited.add(currentState)
            visitDepth[currentState] = depth


            if depth == maxDepth:
                continue

            depth += 1
            for state in self.getSuccessors(currentState):
                if state not in visited or visitDepth[state] > depth:
                    if state.isGoal:
                        return self.ResultGenerator.success(
                            depth=depth,
                            totalStatesCount=totalStatesCount,
                            uniqueStatesCount=len(visited)
                        )

                    stack.put((state, depth))

        return self.ResultGenerator.failure()


    def ids(self):
        startState: Map = self.getStartState()
        if startState.isGoal:
            return self.ResultGenerator.success()

        maxDepth = 1
        res = self.ResultGenerator.failure()
        while not res[self.ResultGenerator.KET_SUCCEEDED]:
            res = self.dls(startState, maxDepth)
            maxDepth += 1

        return res

    def astar(self, heuristic):
        startState: Map = self.getStartState()
        if startState.isGoal:
            return self.ResultGenerator.success()

        queue: PriorityQueue[Map] = PriorityQueue()
        queue.put((heuristic(startState), (startState, 0)))

        visited = set()

        totalStatesCount = 0
        while not queue.empty():
            totalStatesCount += 1

            _, stateDepth = queue.get()
            currentState, depth = stateDepth
            visited.add(currentState)

            depth += 1
            for state in self.getSuccessors(currentState):
                if state not in visited:
                    if state.isGoal:
                        return self.ResultGenerator.success(
                            depth=depth,
                            totalStatesCount=totalStatesCount,
                            uniqueStatesCount=len(visited)
                        )

                    queue.put((heuristic(state) + depth, (state, depth)))

        return self.ResultGenerator.failure()

def h1(state: Map):
    return state.findNearestHospitalToPatientsDistance()

def h2(state: Map):
    return h1(state) * 3

def test(problem):
    from time import time

    print("---------  BFS  ---------")
    start = time()
    print(problem.bfs())
    print("Time: ", time() - start)

    print("---------  IDS  ---------")
    start = time()
    print(problem.ids())
    print("Time: ", time() - start)

    print("---------  A* h1  -------")
    start = time()
    print(problem.astar(h1))
    print("Time: ", time() - start)


    print("---------  A* h2  -------")
    start = time()
    print(problem.astar(h2))
    print("Time: ", time() - start)

    print("\n*************************\n")



test(SearchProblem('./1.in'))
test(SearchProblem('./2.in'))
test(SearchProblem('./3.in'))
