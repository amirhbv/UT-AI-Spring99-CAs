import operator

def moveTuple(pos: tuple, d: tuple) -> tuple:
    return tuple(map(operator.add, pos, d))

def manhattanDistance(a: tuple, b: tuple) -> int:
    x1, y1 = a
    x2, y2 = b
    return abs(x2 - x1) + abs(y2 - y1)

def squaredEuclideanDistance(a: tuple, b: tuple) -> int:
    dx, dy = tuple(map(operator.sub, a, b))
    return dx * dx + dy * dy

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

    def findNearestHospitalDistance(self, patient: tuple, distanceFunction) -> int:
        minDist = -1
        for hospital in self.hospitals:
            dist = distanceFunction(patient, hospital)
            minDist = dist if dist < minDist or minDist == -1 else minDist

        return minDist

    def findPatientsToNearestHospitalDistance(self, distanceFunction) -> int:
        res = 0
        for patient in self.patients:
            res += self.findNearestHospitalDistance(patient, distanceFunction)

        return res

    def findAmbulanceToNearestHospitalDistance(self, distanceFunction) -> int:
        res = 0
        for patient in self.patients:
            res += (distanceFunction(patient, self.ambulance) + self.findNearestHospitalDistance(patient, distanceFunction))

        return res

    @property
    def isGoal(self) -> bool:
        return len(self.patients) == 0

    @classmethod
    def buildMapFromMap(cls, m: Map):
        return cls(
            ambulance=m.ambulance,
            hospitals=m.hospitals,
            patients=m.patients,
            obstacles=m.obstacles
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

from queue import LifoQueue, PriorityQueue, Queue
from typing import List


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

        frontier = Queue()
        frontier.put((startState, 0))

        visited = set()

        totalStatesCount = 0
        while not frontier.empty():
            totalStatesCount += 1

            currentState, depth = frontier.get()
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

                    frontier.put((state, depth))

        return self.ResultGenerator.failure()

    def dls(self, startState: Map, maxDepth: int):
        frontier = LifoQueue()
        frontier.put((startState, 0))

        visited = set()
        visitDepth = {}

        totalStatesCount = 0
        while not frontier.empty():
            totalStatesCount += 1

            currentState, depth = frontier.get()
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

                    frontier.put((state, depth))

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

        frontier = PriorityQueue()
        frontier.put((heuristic(startState), (startState, 0)))

        visited = set()

        totalStatesCount = 0
        while not frontier.empty():
            totalStatesCount += 1

            _, stateDepth = frontier.get()
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

                    frontier.put((heuristic(state) + depth, (state, depth)))

        return self.ResultGenerator.failure()


def h1(state: Map):
    return state.findPatientsToNearestHospitalDistance(manhattanDistance)
    # return state.findPatientsToNearestHospitalDistance(squaredEuclideanDistance)

def h2(state: Map):
    return state.findAmbulanceToNearestHospitalDistance(manhattanDistance)

from time import time
from prettytable import PrettyTable

def test(problem: SearchProblem, repeatCount=3):
    table = PrettyTable()
    table.field_names = [
        'Algorithm',
        SearchProblem.ResultGenerator.KEY_DEPTH,
        SearchProblem.ResultGenerator.KEY_TOTAL_STATES,
        SearchProblem.ResultGenerator.KEY_UNIQUE_STATES,
        'Average Time'
    ]

    totalTime = 0
    for i in range(repeatCount):
        start = time()
        res = problem.bfs()
        totalTime += (time() - start)

    table.add_row([
        'BFS',
        res[SearchProblem.ResultGenerator.KEY_DEPTH],
        res[SearchProblem.ResultGenerator.KEY_TOTAL_STATES],
        res[SearchProblem.ResultGenerator.KEY_UNIQUE_STATES],
        totalTime / repeatCount
    ])

    totalTime = 0
    for i in range(repeatCount):
        start = time()
        res = problem.ids()
        totalTime += (time() - start)

    table.add_row([
        'IDS',
        res[SearchProblem.ResultGenerator.KEY_DEPTH],
        res[SearchProblem.ResultGenerator.KEY_TOTAL_STATES],
        res[SearchProblem.ResultGenerator.KEY_UNIQUE_STATES],
        totalTime / repeatCount
    ])

    totalTime = 0
    for i in range(repeatCount):
        start = time()
        res = problem.astar(h1)
        totalTime += (time() - start)

    table.add_row([
        'A* (h1)',
        res[SearchProblem.ResultGenerator.KEY_DEPTH],
        res[SearchProblem.ResultGenerator.KEY_TOTAL_STATES],
        res[SearchProblem.ResultGenerator.KEY_UNIQUE_STATES],
        totalTime / repeatCount
    ])

    totalTime = 0
    for i in range(repeatCount):
        start = time()
        res = problem.astar(h2)
        totalTime += (time() - start)

    table.add_row([
        'A* (h2)',
        res[SearchProblem.ResultGenerator.KEY_DEPTH],
        res[SearchProblem.ResultGenerator.KEY_TOTAL_STATES],
        res[SearchProblem.ResultGenerator.KEY_UNIQUE_STATES],
        totalTime / repeatCount
    ])

    print(table)


test(SearchProblem('./1.in'))
test(SearchProblem('./2.in'))
test(SearchProblem('./3.in'))
