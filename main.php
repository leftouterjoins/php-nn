<?php

require_once 'nn.php';

$nn = PHP_NN::init('/app/titanic.csv', [
    'loss_function' => function ($n, $p) {
        return ($p - $n) ^ 2;
    },
    'label' => 'Survived',
    'cols' => [
        'Survived' => 'bool',
        'Pclass' => 'string',
        'Sex' => 'string',
        'Age' => 'numeric',
        'Parch' => 'numeric',
        'Fare' => 'numeric',
        'Embarked' => 'string',
        'SibSp' => 'numeric'
    ],
    'normalize' => [
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'Embarked',
    ],
    'hard_params' => [
        'Pclass_1' => -0.04,
        'Pclass_2' => 0.22,
        'Sex_male' => -0.32,
        'Age' => -0.04,
        'Parch' => -0.28,
        'Fare' => 0.10,
        'Embarked_S' => -0.50,
        'Embarked_C' => -0.25,
        'SibSp' => 0.31,
        'Ones' => 0.02
    ],
    'hard_params1' => [
        'Pclass_1' => 0.23,
        'Pclass_2' => -0.21,
        'Sex_male' => 0.18,
        'Age' => 0.35,
        'Parch' => 0.14,
        'Fare' => -0.16,
        'Embarked_S' => 0.22,
        'Embarked_C' => -0.32,
        'SibSp' => -0.01,
        'Ones' => -0.04
    ],
     'logarithmic' => ['Fare']
]);

$nn::train();
