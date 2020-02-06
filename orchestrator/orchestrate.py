from scaleout.runtime.orchestratorruntime import OrchestratorRuntime
from helper import create_seed_model, generate_report, send_report
import time
import numpy as np

""" These routines are for averaging a list of models in Federated Averaging. This code
    is specific to the model type, in this case Keras Sequential. """


class KerasSequentialHelper:

    def average_weights(self,models):
        """ fdfdsfs """
        weights = [model.model.get_weights() for model in models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l,0)
            avg_w.append(weight_l_avg)

        return avg_w

    def set_weights(self,model,weights):
        model.set_weights(weights)


class FederatedAveragingOrchestrator:
    """ Implements Federated Averaging. 'config' should be configurable from the UI (advanced user). """
    def __init__(self, config=None):

        self.global_model_id = None
        self.runtime = OrchestratorRuntime(self)
        

        self.config = config
        if not self.config:
            self.config = {}
            # BEWARE! THIS NUMBER NEEDS TO CORRESPOND TO NUMBER OF TRAINING REQUESTS IN A ROUND, SEE GITHUB API ISSUE
            self.config['round_min_trained_models'] = 1 
            # This (2 mins) is the max we wait in a training round before moving on. 
            self.config['round_timeout'] = 120


        self.round_score = 0.0


    def set_init_model(self, model_id):
        self.global_model_id = model_id
        
    def get_global_model_id(self):
        return self.global_model_id

    def receive_candidate(self, model_id):
        """ Callback when a new model version has been trained. We need to average the 
            weights of all the models updates that are recieved in this round. """ 
        try:
            print("Orchestrator: getting model update with ID {}".format(model_id))
            self.round_models.append(model_id)
        except:
            print("Failed to recieve candidate model!")
            pass

    def __combine_models_and_update_global(self, models):

        print("Orchestrator: averaging models...")
        helper = KerasSequentialHelper()
        model = self.runtime.get_model(models.pop())
        for model_id in models:
            try:
                avg_w = helper.average_weights([model,self.runtime.get_model(model_id)])  
                helper.set_weights(model,avg_w)
            except Exception as e:
                raise(e)
        self.global_model_id=self.runtime.set_model(model)
        print("...done. New global model: {}".format(self.global_model_id))


    def receive_validation(self, model_id, score):
        """ Callback for a validation request """  
        model = {'id': model_id,'score': score}
        self.round_score = score

    def __training_round(self):
        """ One round of training the global model.  """
        self.round_models = []
        self.number_of_trained_round_models = 0
        round_time = 0.0
        status = 0

        # TODO: request training from <member_fraction> number of members in each round 
        # TODO: Get number of requeest and use that instead of round_min_trained_models
        self.runtime.request_contribution()

        # Wait until we have averaged sufficiently many local models in this round
        while len(self.round_models) < self.config['round_min_trained_models']:
            time.sleep(1)
            round_time += 2
            if round_time > self.config['round_timeout']:
                print("Orchestrator: training round timed out.")
                break
            print("Orchestator: training models...{}".format(self.round_models))
 
        # Return sucessful training round if enogh 
        if len(self.round_models) >= self.config['round_min_trained_models']:
            status = 1

        return status

    def __validation_round(self):
        # TODO: Make sure that we only run request validation on the new global model after all 
        # training events has been completed and all models have been averaged. 
        self.round_score = -1
        self.runtime.request_validation(self.global_model_id)
        # No need to wait for validaton to complete before next training round, but do this here to check that it is working...
        timeout = 0
        while self.round_score < 0:
            time.sleep(1)
            timeout += 1
            if timeout > 50:
                break

    def run(self, rounds=10):
        
        for r in range(1, rounds):
            print("STARTING ROUND {}".format(r))
            print("\t Starting training round {}".format(r))
            status = self.__training_round()
            if status:
                print("\t Model training done: {0}".format(self.round_models))
            else:
                print("\t Training round {0} failed!".format(r))

            print("\tAveraging models {}".format(self.round_models))
            self.__combine_models_and_update_global(self.round_models)

            print("\t Starting validation round {}".format(r))
            self.__validation_round()

            global_model = orchestrator.runtime.get_model(orchestrator.global_model_id)
            classification_report = generate_report(global_model)
            print(classification_report)

            # NOTE!!! Model_ID hardcoded, need a way to get that from the SDK. 
            report = {'model':'2','description':'classification_report','report':classification_report,}

            # Need to edit with propos
            status = send_report(self.config['studio_report_endpoint'],report)
            print("Sent report to Studio, {}".format(status))

            print("------------------------------------------")
            print("\tRESULT FROM ROUND {0}, score={1}".format(r,self.round_score))
            print("\n")
 

if __name__ == '__main__':

    #import json
    #with open("orchestrator_config.json","r") as fh:
    #    config = json.loads(fh)
    
    config = {}
    config['round_timeout'] = 120
    config['round_min_trained_models'] = 2
    config['number_of_rounds'] = 5
    config['studio_report_endpoint'] = "http://platform.demo.scaleout.se/projects/andreas/mnist-cnn-alliance/reports/all/"
    
    orchestrator = FederatedAveragingOrchestrator(config=config)

    # Provide a model_id for the seed model. Here we create using a script. 
    # Should also be possible to grab an existing model_id from the db.
    model = create_seed_model()
    seed_model_id = orchestrator.runtime.set_model(model)
    orchestrator.set_init_model(seed_model_id)

    # Start training
    orchestrator.run(rounds=config['number_of_rounds'])

