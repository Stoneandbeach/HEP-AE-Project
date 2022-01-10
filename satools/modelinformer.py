import os

class ModelInformer():
    
    def __init__(self, filename):
        
        self.filename = filename
        if not os.path.isfile(filename):
            print('File', filename, 'doesn\'t exist - creating it.')
            file = open(filename, 'w')
            file.close()
        self.models = self.build_list()
        self.ID_list = self.get_ID_list()
    
    
    
    def build_list(self):
        
        file = open(self.filename, 'r')
        lines = file.readlines()
        file.close()
        
        models = []
        item = {}
        key = ''
        for line in lines:
            line = line.strip('\n')
            if line == '%':
                if not item == {}:
                    models.append(item)
                    item = {}
            else:
                if key == '':
                    key = line
                else:
                    item[key] = line
                    key = ''   
        models.append(item)
        return models
    
    
    
    def get_ID_list(self):
        
        ID_list = []
        for item in self.models:
            ID_list.append(item['modelID'])
        return ID_list
    
    
    
    def get_info(self, modelID = None):
        
        if modelID:
            for model in self.models:
                if model['modelID'] == modelID:
                    return model
            print('Requested modelID not found.')
            return None
        else:
            return self.models
    
    
    
    def save(self, info, modelID = None):
        
        assert 'modelID' in list(info.keys()), 'modelID not found in provided information, aborting.'
        
        if not modelID:
            modelID = info['modelID']
        
        if modelID in self.ID_list:
            print('modelID already in list, aborting.')
        else:                
            file = open(self.filename, 'a')
            file.write('\n%')
            for key in info.keys():
                file.write('\n')
                file.write(key)
                file.write('\n')
                file.write(str(info[key]))
            file.close()
        self.ID_list = self.get_ID_list()
    
    
    
    def get_short(self):
        dc = []
        for item in self.models:
            if 'collective' in item.keys():
                dc.append({'modelID' : item['modelID'], 'trained_on' : item['trained_on'], 'collective' : item['collective']})
            else:
                dc.append({'modelID' : item['modelID'], 'trained_on' : item['trained_on']})
        return dc