// Declare global variables
float startTime = 0;
float endTime = 0;
float executionTime = 0;

// number of items
int n = ...;

// creates range to use in array
range items = 1..n;

// constraints
int profits[items] = ...;
int weights[items] = ...;
int capacity = ...;

// decision variables
dvar boolean x[items];

// start measuring time
execute {
    startTime = cplex.getCplexTime();
}
// linear integer problem
maximize sum(j in items)
    profits[j] * x[j];
subject to
{
  sum(i in items)
    weights[i] * x[i] <= capacity;
}
// stop measuring time
execute {
    endTime = cplex.getCplexTime();
    executionTime = endTime - startTime;
    writeln("Execution time: ", executionTime, " seconds");
}
