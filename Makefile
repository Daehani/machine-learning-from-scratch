.PHONY: all clean

TRAIN_X_PATH=./data/X_train.csv
TRAIN_Y_PATH=./data/y_train.csv
TEST_X_PATH=./data/X_test.csv
TEST_Y_PATH=./data/y_test.csv

all: $(TRAIN_X_PATH) $(TRAIN_Y_PATH) $(TEST_X_PATH) $(TEST_Y_PATH)

clean:
		rm $(TRAIN_X_PATH) $(TRAIN_Y_PATH) $(TEST_X_PATH) $(TEST_Y_PATH)

$(TRAIN_X_PATH) $(TRAIN_Y_PATH) $(TEST_X_PATH) $(TEST_Y_PATH):
	python ./src/make_data.py\
		--save_X_train_path $(TRAIN_X_PATH)\
		--save_y_train_path $(TRAIN_Y_PATH)\
		--save_X_test_path $(TEST_X_PATH)\
		--save_y_test_path $(TEST_Y_PATH)