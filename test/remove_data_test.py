from citl.dataset import Dataset
import os 

current_path = os.path.dirname(os.path.realpath(__file__))

def test_remove_data():
    augmentation_policy_path = os.path.join(current_path, "..", "policies", "noop.yaml")
    dm = Dataset.get("CIFAR10")(augmentation_policy_path)
    dm.set_image_size(32, True)
    dm.setup()
    first_item = dm.cifar_train.indices[0]
    second_item = dm.cifar_train.indices[1]
    assert len(dm.cifar_train) == 45000
    dm.remove_train_example(0)
    assert len(dm.cifar_train) == 44999
    assert first_item not in dm.cifar_train.indices
    assert dm.cifar_train.indices[0] == second_item
