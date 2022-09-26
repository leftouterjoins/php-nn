<?php declare(strict_types = 1);

class PHP_NN
{
    protected const CATEGORY_MAX = 3;

    private static ?self $singleton;

    protected string $trainingDataPath;
    protected array $options;

    protected array $params;
    protected array $params1;

    protected array $dataframe;
    protected array $dataframeNrml;

    protected array $labelData;

    protected array $cols;
    protected array $normalize;
    protected Closure $lossFunction;

    public static function init(string $trainingDataPath, array $options): self
    {
        if (empty(self::$singleton)) {
            self::$singleton = new self;
            self::$singleton->trainingDataPath = $trainingDataPath;
            self::$singleton->options = $options;
         }

        return self::$singleton;
    }

    public static function train(): void
    {
        self::$singleton->unpack(self::$singleton->options);
        self::$singleton->loadTrainingData();
        self::$singleton->normalize();
        self::$singleton->initParams();

        $loss = self::$singleton->calculate(self::$singleton->params, self::$singleton->params1);

        $step = -0.01;
        $gradient = function () use ($step) {

            # Get last params used.
            $newParams = self::$singleton->params;
            $newParams1 = self::$singleton->params1;

            # Increment each param by step.
            foreach (array_keys($newParams) as $key) {
                $newParams[$key] = $newParams[$key] + $step;
                $newParams1[$key] = $newParams1[$key] + $step;
            }

            # Set new params as last params used.
            self::$singleton->params = $newParams;
            self::$singleton->params1 = $newParams1;

            # Calculate new params.
            return self::$singleton->calculate($newParams, $newParams1);
        };

        $min = self::$singleton->descendGradient(
            $loss,
            $gradient,
            $step,
            count(self::$singleton->dataframe),
            0.01
        );

        var_dump($min);
    }

    protected function calculate(array $params, array $params1)
    {
        $predictions = $this->clip($this->sumProduct(
            $params,
            $this->dataframeNrml
        ));

        $predictions1 = $this->clip($this->sumProduct(
            $params1,
            $this->dataframeNrml
        ));

        foreach ($predictions as $serial => $prediction) {
            $predictions[$serial] = $prediction + $predictions1[$serial];
        }

        $loss = $this->loss($predictions);
        $lossAvg = array_sum($loss) / count($loss);

        return $lossAvg;
    }

    protected function descendGradient($start, $gradient, $step, $count, $threshold): float
    {
        $steps = [$start];
        $x = $start;

        for ($i = 0; $i < $count; $i++) {
            $loss = $gradient();
            $diff = $step * $loss;

            if (abs($diff) < $threshold) {
                break;
            }

            $x -= $diff;
            $steps[] = $x;
        }

        return $loss;
    }

    protected function sumProduct($params, $data)
    {
        $sums = [];
        foreach ($data as $serial => $row) {
            $sum = 0.0;
            foreach ($params as $col => $param) {

                $sum += $param * $row[$col];
            }

            $sums[$serial] = $sum;
        }

        return $sums;
    }

    protected function unpack($p): void
    {
        $this->cols = $p['cols'];
        $this->lossFunction = $p['loss_function'];
        $this->normalize = $p['normalize'];
    }

    protected function loadTrainingData(): array
    {
        if (empty($this->dataframe)) {

            $this->dataframe = [];
        } else {

            return $this->dataframe;
        }

        $fh = fopen($this->trainingDataPath, 'r');

        $cols = fgetcsv($fh);
        while ($row = fgetcsv($fh)) {
            $this->dataframe[] = array_combine($cols, $row);
        }
        /*

        $this->dataframe = array_slice(
            $this->dataframe,
            659,
            53,
            true
        );*/

        return $this->dataframe;
    }

    protected function categorize()
    {
        $categories = [];

        foreach ($this->normalize as $col) {

            unset($range, $type); # reset

            $colData = array_column($this->dataframe, $col);
            $uniqVals = array_filter(array_unique($colData));

            if (!is_numeric($colData[0]) || count($uniqVals) <= self::CATEGORY_MAX) {
                $type = 'categorical';
            } elseif (is_numeric($colData[0])) {
                $type = 'numeric';
            } else {
                continue;
            }

            if (is_numeric($colData[0])) {
                sort($uniqVals);
            }

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

    protected function normalize()
    {
        $categories = $this->categorize();

        foreach ($this->dataframe as $index => $row) {
            $this->dataframeNrml[$index] = [];

            $this->labelData[$index] = $row[$this->options['label']];
            unset($row[$this->options['label']]);

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
                            $this->dataframeNrml[$index][$col . "_$opt"] = 0.0;
                        }

                        if (in_array($value, $range)) {
                            $this->dataframeNrml[$index][$col . "_$value"] = 1.0;
                        }
                    }

                    if ('numeric' === $type) {

                        if (in_array($col, $this->options['logarithmic'])) {
                            $value = log10(floatval($value)+1);
                        } else {
                            $value = floatval($value) / $range;
                        }

                        $this->dataframeNrml[$index][$col] = $value;
                    }

                    continue;
                }

                $datatype = $this->cols[$col];

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

                $this->dataframeNrml[$index][$col] = $value;
            }

            $this->dataframeNrml[$index]['Ones'] = 1.0;
        }
    }

    protected function initParams()
    {
        if (array_key_exists('hard_params', $this->options)) {

            $this->params = $this->options['hard_params'];
            $this->params1 = $this->options['hard_params1'];

            return;
        }

        $params = $this->dataframeNrml[0];
        $params1 = $this->dataframeNrml[0];

        foreach (array_keys($params) as $col) {
            $params[$col] = (mt_rand(0, 1) - 0.5) / 100;
            $params1[$col] = (mt_rand(0, 1) - 0.5) / 100;
        }

        $this->params = $params;
        $this->params1 = $params1;
    }

    protected function clip(array $n)
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

    protected function loss($predictions)
    {
        $loss = [];
        foreach ($predictions as $serial => $prediction) {
            $label = floatval($this->labelData[$serial]);
            $loss[$serial] = pow(($prediction - $label), 2);
        }

        return $loss;
    }
}
