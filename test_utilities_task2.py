import unittest
from unittest.mock import MagicMock
from unittest import mock
import pickle
import numpy
import utilities_task2
import scipy.special


class UtilitiesTask2Test(unittest.TestCase):
    

    def test_task2_load_cases(self):
        x_train, y_train, x_val, y_val = utilities_task2.task2_load_cases("textf")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 956)

        file = open('./features/dataset2/par_textf_train.pickle', "rb")
        features_textf = pickle.load(file)
        file.close()

        number_of_paragraph_combinations = 0
        for doc in features_textf:
            number_of_paragraph_combinations += (scipy.special.binom(len(doc),2))
        self.assertEqual(number_of_paragraph_combinations, len(x_train))

        x_train, y_train, x_val, y_val = utilities_task2.task2_load_cases("emb")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 768)

        file = open('./features/dataset2/par_emb_train.pickle', "rb")
        features_emb = pickle.load(file)
        file.close()
        number_of_paragraph_combinations = 0
        for doc in features_emb:
            number_of_paragraph_combinations += (scipy.special.binom(len(doc),2))
        self.assertEqual(number_of_paragraph_combinations, len(x_train))

        self.assertEqual(len(features_textf), 7000)
        self.assertEqual(len(features_emb), 7000)
    
       
    @mock.patch('utilities_task2.PAR_EMB_TRAIN_FOR_TASK2', './features/dataset2/file_which_does_not_exist')
    def test_task2_load_cases_errors(self):
        with self.assertRaises(ValueError):
            utilities_task2.task2_load_cases("test")

        with self.assertRaises(OSError):
            utilities_task2.task2_load_cases("emb")


    def test_my_task2_predicitons_emb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task2_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4),(0.2,0.8),(0.6,0.4),(0.1,0.9)]))
        task2_model.predict_proba = predict_proba
        predictions = utilities_task2.my_task2_binary_predictions_emb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertTrue(predictions[0] == 0.4 and predictions[1] == 0.7)
        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,1,2,3]]) )

        predict=MagicMock(return_value=([(0.6),(0.3),(0.6),(0.2),(0.6),(0.1)]))
        task2_model.predict = predict
        predictions = utilities_task2.my_task2_binary_predictions_emb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertTrue(predictions[0] == 0.6 and predictions[1] == 0.3)

        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,2,1,1]]) )

        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        predictions =[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))

        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,2,3,4,5,1]]) )


        
        
        with self.assertRaises(AssertionError):
            predictions = utilities_task2.my_task2_binary_predictions_emb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            predictions = utilities_task2.my_task2_binary_predictions_emb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=True)

    def test_my_task2_predicitons_textf(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task2_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4),(0.2,0.8),(0.6,0.4),(0.1,0.9)]))
        task2_model.predict_proba = predict_proba
        predictions = utilities_task2.my_task2_binary_predictions_textf(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertTrue(predictions[0] == 0.4 and predictions[1] == 0.7)

        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,1,2,3]]) )

        predict=MagicMock(return_value=([(0.6),(0.3),(0.6),(0.2),(0.6),(0.1)]))
        task2_model.predict = predict
        predictions = utilities_task2.my_task2_binary_predictions_textf(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertTrue(predictions[0] == 0.6 and predictions[1] == 0.3)

        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,2,1,1]]) )
        
        with self.assertRaises(AssertionError):
            predictions = utilities_task2.my_task2_binary_predictions_textf(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            predictions = utilities_task2.my_task2_binary_predictions_textf(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=True)


    def test_my_task2_predicitons_comb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task2_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4),(0.2,0.8),(0.6,0.4),(0.1,0.9)]))
        task2_model.predict_proba = predict_proba
        predictions = utilities_task2.my_task2_binary_predictions_comb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        self.assertTrue(predictions[0] == 0.4 and predictions[1] == 0.7)

        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,1,2,3]]) )

        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])
        task2_model.predict_proba = MagicMock(return_value=([0.4,0.6]))
        predictions = utilities_task2.my_task2_binary_predictions_comb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=False, stacking=True)
        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,1],[1,2]]) )
        

        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])
        predict=MagicMock(return_value=([(0.6),(0.3),(0.6),(0.2),(0.6),(0.1)]))
        task2_model.predict = predict
        predictions = utilities_task2.my_task2_binary_predictions_comb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        self.assertTrue(predictions[0] == 0.6 and predictions[1] == 0.3)

        final_pred = utilities_task2.my_task2_final_authorship_predictions(task2_binary_preds=predictions, par_emb=features,par_textf=features)
        self.assertEqual(len(final_pred), len(features))
        numpy.testing.assert_array_equal(final_pred,numpy.array([[1,2,1,1]]) )

        with self.assertRaises(AssertionError):
            predictions = utilities_task2.my_task2_binary_predictions_comb(task2_model=task2_model, par_emb=features, par_textf=features, lgb=True, stacking=True)



        



if __name__ == '__main__':
    unittest.main()
