import operator
from queue import LifoQueue, PriorityQueue, Queue
from random import shuffle
from typing import List, Set
from time import time


def moveTuple(pos: tuple, d: tuple) -> tuple:
    return tuple(map(operator.add, pos, d))


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

    def __hash__(self):
        return hash((
            self.ambulance,
            tuple(self.hospitals.items()),
            tuple(self.patients)
        ))

class SearchProblem:
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
            return 0

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
                        print("-------------------------")
                        print("Total states: ", totalStatesCount)
                        print("Unique states: ", len(visited))
                        print("Depth: ", depth)
                        print("-------------------------")
                        return depth

                    queue.put((state, depth))

        return -1


problem = SearchProblem('./1.in')

print(time())
problem.bfs()
print(time())
