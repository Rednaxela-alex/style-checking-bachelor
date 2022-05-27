import unittest
from unittest.mock import MagicMock
from unittest import mock
import pickle
import numpy
import torch
import utilities_task1
import scipy.special


class UtilitiesTask1Test(unittest.TestCase):
    def __len__(self):
        return 2

    def test_task1_load_cases(self):
        x_train, y_train, x_val, y_val = utilities_task1.task1_load_cases("textf")
        self.assertEqual(numpy.sum(y_train), 1400)
        self.assertEqual(numpy.sum(y_val), 300)
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 956)

        file = open('./features/dataset1/par_textf_train.pickle', "rb")
        features_textf = pickle.load(file)
        file.close()

        x_train, y_train, x_val, y_val = utilities_task1.task1_load_cases("emb")
        self.assertEqual(numpy.sum(y_train), 1400)
        self.assertEqual(numpy.sum(y_val), 300)
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 768)

        file = open('./features/dataset1/par_emb_train.pickle', "rb")
        features_emb = pickle.load(file)
        file.close()
        number_of_paragraph_combinations = 0
        for doc in features_emb:
            number_of_paragraph_combinations += len(doc)-1
        self.assertEqual(number_of_paragraph_combinations, len(x_train))
        
        self.assertEqual(len(features_textf), 1400)
        self.assertEqual(len(features_emb), 1400)
    
    def test_task1_load_cases_comparing_each_paragraph(self):
        x_train, y_train, x_val, y_val = utilities_task1.task1_load_cases_comparing_each_paragraph("textf")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 956)

        x_train, y_train, x_val, y_val = utilities_task1.task1_load_cases_comparing_each_paragraph("emb")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 768)

        file = open('./features/dataset1/par_emb_train.pickle', "rb")
        features_emb = pickle.load(file)
        file.close()
        number_of_paragraph_combinations = 0
        for doc in features_emb:
            number_of_paragraph_combinations += (scipy.special.binom(len(doc),2))
        self.assertEqual(number_of_paragraph_combinations, len(x_train))

    

    @mock.patch('utilities_task1.PAR_EMB_TRAIN_FOR_TASK1', './features/dataset1/file_which_does_not_exist')
    def test_task1_load_cases_errors(self):
        with self.assertRaises(ValueError):
            utilities_task1.task1_load_cases("test")

        with self.assertRaises(ValueError):
            utilities_task1.task1_load_cases_comparing_each_paragraph("test")

        with self.assertRaises(OSError):
            utilities_task1.task1_load_cases("emb")

        with self.assertRaises(OSError):
            utilities_task1.task1_load_cases_comparing_each_paragraph("emb")


    def test_my_task1_parchange_predicitons_emb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task1_model = MagicMock()
        
        predict_proba=MagicMock(return_value=([(1,0),(0,1)]))
        predict_proba.__len__ = self.__len__
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 0 and prediction[1] == 1)

        predict_proba=MagicMock(return_value=([(0,1),(1,0)]))
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 1 and prediction[1] == 0)

        predict_proba=MagicMock(return_value=([(1,0),(1,0)]))
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 1 and prediction[1] == 0)

        predict_proba=MagicMock(return_value=([(0,1),(0,1)]))
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 1 and prediction[1] == 0)
        
        predict=MagicMock(return_value=(numpy.array([1,0])))
        task1_model.predict = predict
        predictions = utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)


        
        self.assertEqual(len(features), len(predictions))

        with self.assertRaises(AssertionError):
            utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=True)


    def test_my_task1_parchange_predicitons_textf(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task1_model = MagicMock()
        
        predict_proba=MagicMock(return_value=([(1,0),(0,1)]))
        predict_proba.__len__ = self.__len__
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_textf(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 0 and prediction[1] == 1)
        
        predict=MagicMock(return_value=(numpy.array([1,0])))
        task1_model.predict = predict
        predictions = utilities_task1.my_task1_parchange_predictions_textf(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)

        with self.assertRaises(AssertionError):
            utilities_task1.my_task1_parchange_predictions_textf(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            utilities_task1.my_task1_parchange_predictions_textf(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=True)


    def test_my_task1_parchange_predicitons_comb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task1_model = MagicMock()
        
        predict_proba=MagicMock(return_value=([(1,0),(0,1)]))
        predict_proba.__len__ = self.__len__
        task1_model.predict_proba = predict_proba
        predictions = utilities_task1.my_task1_parchange_predictions_comb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)
            self.assertTrue(prediction[0] == 0 and prediction[1] == 1)

        
        predict=MagicMock(return_value=(numpy.array([1,0])))
        task1_model.predict = predict
        predictions = utilities_task1.my_task1_parchange_predictions_comb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertEqual(len(predictions), len(features))
        for prediction in predictions:
            self.assertTrue(torch.sum(prediction) == 1)

        with self.assertRaises(AssertionError):
            utilities_task1.my_task1_parchange_predictions_emb(task1_model=task1_model, par_emb=features, par_textf=features, lgb=True, stacking=True)





if __name__ == '__main__':
    unittest.main()
