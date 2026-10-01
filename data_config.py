
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    image_type="png"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '.../LEVIR-CD-256/'
        elif data_name == 'LEVIR+':
            self.label_transform = "norm"
            self.root_dir = '.../LEVIR-CD+-256/'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '.../WHU-CD-256/'
        elif data_name == 'EGY_BCD':
            self.label_transform = "norm"
            self.root_dir = '.../EGY-BCD-256/'
        elif data_name =='Syntheworld':
            self.label_transform = "norm"
            self.root_dir = '.../syntheworld-512-256_1M/'
        elif data_name =='GZ':
            self.label_transform = "norm"
            self.root_dir = '.../GZ-CD/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

