import unittest
from unittest.mock import MagicMock
from unittest import mock
import pickle
import numpy
import utilities_task3


class UtilitiesTask3Test(unittest.TestCase):
    

    def test_task3_load_cases(self):
        x_train, y_train, x_val, y_val = utilities_task3.task3_load_cases("textf")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 956)

        file = open('./features/dataset3/par_textf_train.pickle', "rb")
        features_textf = pickle.load(file)
        file.close()

        number_of_paragraphs = 0
        for doc in features_textf:
            number_of_paragraphs += (len(doc)-1)
        self.assertEqual(number_of_paragraphs, len(x_train))

        x_train, y_train, x_val, y_val = utilities_task3.task3_load_cases("emb")
        self.assertEqual(len(x_val[0]), len(x_train[0]))
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_val), len(y_val))
        self.assertEqual(len(x_val[0]), 768)

        file = open('./features/dataset3/par_emb_train.pickle', "rb")
        features_emb = pickle.load(file)
        file.close()
        number_of_paragraphs = 0
        for doc in features_emb:
            number_of_paragraphs += len(doc)-1
        self.assertEqual(number_of_paragraphs, len(x_train))

        self.assertEqual(len(features_textf), 7000)
        self.assertEqual(len(features_emb), 7000)
    
       
    @mock.patch('utilities_task3.PAR_EMB_TRAIN_FOR_TASK3', './features/dataset3/file_which_does_not_exist')
    def test_task3_load_cases_errors(self):
        with self.assertRaises(ValueError):
            utilities_task3.task3_load_cases("test")

        with self.assertRaises(OSError):
            utilities_task3.task3_load_cases("emb")


    def test_my_task3_predicitons_emb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task3_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4)]))
        predict = MagicMock(return_value=([0.4,0.7,0.2]))
        task3_model.predict_proba = predict_proba
        task3_model.predict = predict
        predictions = utilities_task3.my_task3_parchange_predictions_emb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        lgb_predictions = utilities_task3.my_task3_parchange_predictions_emb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        
        self.assertEqual(len(predictions), len(features))
        numpy.testing.assert_array_equal(predictions,numpy.array([[0,1,0]]) )
        numpy.testing.assert_array_equal(lgb_predictions,numpy.array([[0,1,0]]) )


        

        
        


        
        
        with self.assertRaises(AssertionError):
            predictions = utilities_task3.my_task3_parchange_predictions_emb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            predictions = utilities_task3.my_task3_parchange_predictions_emb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=True)

    def test_my_task3_predicitons_textf(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task3_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4)]))
        predict = MagicMock(return_value=([0.4,0.7,0.2]))
        task3_model.predict_proba = predict_proba
        task3_model.predict = predict
        predictions = utilities_task3.my_task3_parchange_predictions_textf(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        lgb_predictions = utilities_task3.my_task3_parchange_predictions_textf(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        numpy.testing.assert_array_equal(predictions,numpy.array([[0,1,0]]) )
        numpy.testing.assert_array_equal(lgb_predictions,numpy.array([[0,1,0]]) )

        self.assertEqual(len(predictions), len(features))
        
        with self.assertRaises(AssertionError):
            predictions = utilities_task3.my_task3_parchange_predictions_textf(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=True)

        with self.assertRaises(AssertionError):
            predictions = utilities_task3.my_task3_parchange_predictions_textf(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=True)


    def test_my_task3_predicitons_comb(self):
        
        features = numpy.array([[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]])       
        task3_model = MagicMock()
        #the mocked probabilities are the probs for (no style change, style change)
        #and the order for paracombinations in this example is (1/2,1/3,2/3,1/4,2/4,3/4)
        predict_proba=MagicMock(return_value=([(0.6,0.4),(0.3,0.7),(0.6,0.4)]))
        predict = MagicMock(return_value=[0.4,0.7,0.2])
        task3_model.predict_proba = predict_proba
        task3_model.predict = predict
        predictions = utilities_task3.my_task3_parchange_predictions_comb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=False)
        lgb_predictions = utilities_task3.my_task3_parchange_predictions_comb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=False)
        numpy.testing.assert_array_equal(predictions,numpy.array([[0,1,0]]) )
        numpy.testing.assert_array_equal(lgb_predictions,numpy.array([[0,1,0]]) )
        
        self.assertEqual(len(predictions), len(features))
        task3_model.predict_proba = MagicMock(return_value=[0.4,0.7,0.2])
        predictions = utilities_task3.my_task3_parchange_predictions_comb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=False, stacking=True)
        numpy.testing.assert_array_equal(predictions,numpy.array([[0,1,0]]) )
        self.assertEqual(len(predictions), len(features))

        with self.assertRaises(AssertionError):
            predictions = utilities_task3.my_task3_parchange_predictions_comb(task3_model=task3_model, par_emb=features, par_textf=features, lgb=True, stacking=True)



        



if __name__ == '__main__':
    unittest.main()
