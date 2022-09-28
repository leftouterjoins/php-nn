<?php declare(strict_types = 1);

$trainCsv = './titanic-train.csv';
$testCsv  = './titanic-test.csv';

$setup = [

    'cat_max' => 3,           # The maximum number of catgegories allowed in a categorical column. (can baloon parameter size!)
    'iterations' => 10000,    # How many iterations to run along the gradient.
    'label' => 'Survived',    # Which column contains the the result we are trying to predict.
    'learning_rate' => 0.1,   # How fast should we descend the gradient?

    'cols' => [               # Select the relevant columns from the CSV file and cast them.
        'Survived' => 'bool',
        'Pclass' => 'string',
        'Sex' => 'string',
        'Age' => 'numeric',
        'Parch' => 'numeric',
        'Fare' => 'numeric',
        'Embarked' => 'string',
        'SibSp' => 'numeric'
    ],

    'normalize' => [          # Normalize these columns using numerical and categorical means.
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'Embarked',
    ],

    'logarithmic' => ['Fare'], # Use logarthimic normalization for these numeric fields. (Good for $$ data)

    'subset' => [658, 55]      # In the fastai course video 3 this is the subset of data used (approximately)
];

class LearningMachine
{
    # Data

    protected array $dataframe;       # original dataframe
    protected array $labelData;       # label data (actual outcomes)
    protected array $params;          # model parameters

    # User Settings

    public array   $cols;             # These are the names of the columns in the CSV file that we care about.
    public int     $iterations;       # How many times to run the training loop.
    public float   $learningRate;     # How fast to train the network.
    public array   $logarithmic;      # Which columns to apply a logarithmic function to.
    public array   $normalize;        # Which columns to normalize.
    public int     $catMax = 3;       # The maximum number of uniq values in a column for it to be considered a categorical.
    public string  $label;            # The name of the column that contains the label data.
    public string  $trainingDataPath; # The path to the CSV file containing the training data.
    public string  $testingDataPath;  # The path to the CSV file containing the testing data.
    public ?array  $subset;           # The number of rows to use for training and testing.

    public function __construct(string $trainingDataPath, string $testingDataPath)
    {
        $this->trainingDataPath = $trainingDataPath;
        $this->testingDataPath  = $testingDataPath;
    }

    public function run(array $options): float
    {
        $this->unpack($options);                   # Hydrate object properties with the passed options.

        $this->loadTrainingData();                 # Load the CSV file into memory.
        $this->normalize();                        # Normalize the data.

        $paramCnt = count($this->dataframe[0]);    # How many parameters do we need to train?
        echo "with $paramCnt parameters...\n\t";

        $this->params = $this->initParams();       # Initialize the model parameters.
        $loss         = $this->descendGradient();  # Run the training loop.

        # Report final parameters and loss.
        echo "loss was $loss\n";

        return $loss;
    }

    public function infer()
    {
        $this->dataframe = [];
        $this->loadTestingData();
        $this->normalize();

        $predictions = $this->makePredictions($this->params, $this->dataframe);

        foreach ($predictions as $i => $prediction) {
            echo "Passenger $i: " . ($prediction > 0.05 ? 'Survived' : 'Died') . "\n";
        }
    }

    public function unpack($p): void
    {
        $this->learningRate = $p['learning_rate'];
        $this->iterations   = $p['iterations'];
        $this->label        = $p['label'];
        $this->cols         = $p['cols'];
        $this->normalize    = $p['normalize'];
        $this->catMax       = $p['cat_max'];
        $this->logarithmic  = $p['logarithmic'];
        $this->subset       = $p['subset'] ?? null;
    }

    public function initParams(): array
    {
        $params = $this->dataframe[0]; # Peel off the first row to use as a parameter template.

        # Assign a random value to each parameter.

        foreach (array_keys($params) as $col) {
            $params[$col] = mt_rand(0, 1) / 100;
        }

        return $params;
    }

    public function descendGradient(): float
    {
        # Run the loop for the specified number of iterations.

        for ($i = 0; $i < $this->iterations; $i++) {

            # Train the model on each row of the dataframe.

            $loss = $this->train($this->params);

            # Adjust the parameters down gradient so we can try again.

            $this->params = $this->adjustParams($this->params);
        }

        return $loss;
    }

    protected function loss(array $predictions): array
    {
        # Calculate the loss for each prediction.

        $loss = [];
        foreach ($predictions as $serial => $prediction) {
            $label = $this->labelData[$serial];           # The actual outcome.
            $loss[$serial] = ($prediction - $label) ** 2; # Mean Squared Error
        }

        return $loss;
    }

    protected function adjustParams($params): array
    {
        # Copy params so we can adjust them without affecting the original.

        $adjParams = $params;

        foreach ($params as $param => $value) {

            # Calculate the partial derivative of the loss function with respect to
            # the parameter and adjust it using the learning rate.

            $p = $value - $this->learningRate * $this->partialDerivative($param);
            $adjParams[$param] = $p;
        }

        # Return the adjusted parameters.

        return $adjParams;
    }

    protected function partialDerivative(string $param): float
    {
        #
        # TBH, this is the part I understand the least. Need to read up more on derivatives and calculus. The following
        # is the best of my understanding as is likely to be wrong. I'm just going to leave it here for now.
        #
        # We need to calculate the derivative of the loss function to get the slope of the gradient at the current point.
        # Because we have a multivariate function (i.e. we have more than one parameter in our model), we need to calculate
        # the partial derivative of the loss function. Which is the same as a derivative, but we only consider
        # one dimension or parameter at a time. The sum of the partial derivatives is used to calculate the final
        # derivative of the loss function.
        #
        # The exact formula for calculating a derivative is based on the loss function being used. In this case, we are
        # using the mean squared error loss function. The formula used below I pieced together through a lot of research
        # and trial and error. I am not 100% sure it is correct, but the numbers seem to work out correctly.
        #
        # If I had to do this for a different loss function, I would need to research the correct formula for it or just
        # actually learn calculus. #shudder
        #
        # #IDidntGoToUni
        #

        # Calculate the partial derivative of the loss function with respect to the parameter.

        $results =  [];
        foreach ($this->dataframe as $i => $row) {

            $truth = $this->labelData[$i]; # The actual outcome.
            $results[$i] = ($truth - $this->predict($this->params, $row)) * $row[$param];
        }

        $rowCount = count($this->dataframe);

        return (-2 / $rowCount) * array_sum($results);
    }

    protected function train(array $params): float
    {
        # Make our predictions using the current parameters.

        $predictions = $this->makePredictions(
            $params,
            $this->dataframe
        );

        # Calculate the loss for each prediction.
        $loss = $this->loss($predictions);

        # Return the average loss.

        $lossAvg = array_sum($loss) / count($loss);

        return $lossAvg;
    }

    protected function predict(array $params, array $row): float
    {
        # Multiply each parameter by the corresponding value in the row to make a prediction.

        $sum = 0.0;
        foreach ($params as $col => $param) {
            $sum += $param * $row[$col];
        }

        return $sum;
    }

    protected function makePredictions(array $params, array $data): array
    {
        # Sum the predictions from each row.

        $sums = [];
        foreach ($data as $serial => $row) {
            $sums[$serial] = $this->predict($params, $row);
        }

        return $sums;
    }

    # Boilerplate methods to load and process data from CSV.

    protected function categorize(): array
    {
        $categories = [];

        foreach ($this->normalize as $col) {

            unset($range, $type); # reset

            $colData = array_column($this->dataframe, $col);
            $uniqVals = array_filter(array_unique($colData));


            if (empty($colData)) {
                var_dump($this->dataframe);
                debug_print_backtrace();
                die();
            }

            if (!is_numeric($colData[0]) || count($uniqVals) <= $this->catMax) {
                $type = 'categorical';
            } elseif (is_numeric($colData[0])) {
                $type = 'numeric';
            } else {
                continue;
            }

            sort($uniqVals);

            array_pop($uniqVals);

            if ('categorical' === $type) {

                $range = $uniqVals;

            } elseif ('numeric' === $type) {

                $range = max($colData);
            }

           $categories[$col]['type'] = $type ?? null;
           $categories[$col]['range'] = $range ?? null;
        }

        return $categories;
    }

    public function loadTestingData(): array
    {
        return $this->loadData($this->testingDataPath, false);
    }

    public function loadTrainingData(): array
    {
        return $this->loadData($this->trainingDataPath);
    }

    public function loadData(string $path, bool $subset = true): array
    {
         if (empty($this->dataframe)) {

            $this->dataframe = [];
        } else {

            return $this->dataframe;
        }

        $fh = fopen($path, 'r');

        $cols = fgetcsv($fh);
        while ($row = fgetcsv($fh)) {
            $this->dataframe[] = array_combine($cols, $row);
        }

        if ($subset && !is_null($this->subset)) {
            $this->dataframe = array_slice($this->dataframe, $this->subset[0], $this->subset[1]);
        }

        return $this->dataframe;
    }

    public function normalize(): void
    {
        $categories = $this->categorize();

        $dataframe = [];
        foreach ($this->dataframe as $index => $row) {

            $dataframe[$index] = [];

            $this->labelData[$index] = $row[$this->label] ?? null;
            unset($row[$this->label]);

            $dataframe[$index]['Ones'] = 1.0;

            foreach ($row as $col => $value) {

                if (!in_array($col, array_keys($this->cols))) {
                    continue;
                }

                # Is this a normalized column?
                if (in_array($col, array_keys($categories))) {

                    $category = $categories[$col];
                    $type = $category['type'];
                    $range = $category['range'];

                    if ('categorical' === $type) {

                        foreach ($range as $opt) {
                            $dataframe[$index][$col . "_$opt"] = 0.0;
                        }

                        if (in_array($value, $range)) {
                            $dataframe[$index][$col . "_$value"] = 1.0;
                        }
                    }

                    if ('numeric' === $type) {

                        if (in_array($col, $this->logarithmic)) {
                            $value = log10(floatval($value)+1);
                        } else {
                            $value = floatval($value) / $range;
                        }

                        $dataframe[$index][$col] = $value;
                    }

                    continue;
                }

                $datatype = $this->cols[$col];
                $value =  $this->cast($datatype, $value);

                $dataframe[$index][$col] = $value;
            }

        }

        $this->dataframe = $dataframe;
    }

    protected function cast(string $datatype, mixed $value): mixed
    {
        switch ($datatype) {
            case 'bool':
                $value = boolval($value);
                break;
            case 'numeric':
                $value = floatval($value);
                break;
            case 'string':
                $value = strval($value);
                break;
        }

        return $value;
    }
}

class NeuralLearningMachine extends LearningMachine
{
    protected function predict(array $layers, array $row): float
    {
        $sums = [];

        for ($i = 0; $i < 2; $i++) {
            $sums[$i] = parent::predict($this->params[$i], $row);
        }

        $sums = $this->clip($sums);

        return array_sum($sums);
    }

    protected function adjustParams($params): array
    {
        $adjParams = $params;

        foreach ($params as $layer => $layerParams) {
            $adjParams[$layer] = parent::adjustParams($layerParams);
        }

        return $adjParams;
    }

    public function initParams(): array
    {
        $params = [];
        for ($i = 0; $i < 2; $i++) {
            $params[$i] = parent::initParams();

        }

        $this->params = $params;

        return $this->params;
    }

    protected function clip(array $n): array
    {
        $o = [];
        foreach ($n as $k => $x) {
            if ($x < 0) {
                $x = 0;
            }

            $o[$k] = $x;
        }

        return $o;
    }
 }

echo "Running Linear Regression ";
$linLoss = (new LearningMachine($trainCsv, $testCsv))->run($setup);

echo "Running Neural Network ";
$nn = new NeuralLearningMachine($trainCsv, $testCsv);
$neurLoss = $nn->run($setup);
$improvement = round(($linLoss - $neurLoss) * 100, 2);

echo "Neural network was $improvement% more accurate than linear regression.\n";

$accuracy = round((1 - $neurLoss) * 100, 2);
echo "Model can make predictions with approximately $accuracy% accuracy.\n";

$nn->infer();


