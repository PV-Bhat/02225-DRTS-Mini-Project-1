class Task:
    def __init__ (self, task_id, bcet, wcet, period, deadline, priority):
        self.task_id = task_id
        self.bcet = bcet        #Best Case Execution Time
        self.wcet = wcet        #Worst Case Execution Time
        self.period = period
        self.deadline = deadline
        self.priority = None    #For DM/RM 