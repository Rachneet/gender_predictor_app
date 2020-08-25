import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';

class App extends Component {

   constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      formData: {
        text:""
      },
      result: "",
      errors: []
    };
  }


  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData
    });
  }

  hasError(key) {
    return this.state.errors.indexOf(key) !== -1;
  }

  handlePredictClick = (event) => {
     event.preventDefault();
     var errors = []
    const formData = this.state.formData;

     // text
    if (formData.text === "") {
      errors.push("text");
    }
    else {
      this.setState({isLoading: true});
      fetch('http://127.0.0.1:5000/prediction/',
          {
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify(formData)
          })
          .then(response => response.json())
          .then(response => {
            this.setState({
              result: response.result,
              isLoading: false
            });
          });
    }

    this.setState({
      errors: errors
    });

    if (errors.length > 0) {
      return false;
    } else {
      alert("Everything good. Submit text!");
    }
  }

  handleCancelClick = (event) => {
      const formData = this.state.formData;
      formData.text = ""
    this.setState({formData, result: ""});
  }


  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    const result = this.state.result;

    return (
     <Container>
        <div>
          <h1 className="title">Gender Predictor</h1>
        </div>
        <div>
          <Form>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Text</Form.Label>
                <Form.Control
                    className={
                this.hasError("text")
                ? "form-control is-invalid"
                : "form-control"
            }

                  type="text"
                  placeholder="Enter text"
                  name="text"
                  value={formData.text}
                  as="textarea"
                  rows="4"
                  onChange={this.handleChange} />

              </Form.Group>
            </Form.Row>

             <div
            className={
              this.hasError("text") ? "inline-errormsg" : "hidden"
            }
          >
            Please enter a text
          </div>


            <Row className={"pred-button"}>
              <Col >
                <Button
                    size={"sm"}
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}>
                  { isLoading ? 'Making prediction' : 'Predict' }
                </Button>
              </Col>

              <Col >
                <Button size={"sm"}
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}>
                  Reset prediction
                </Button>
              </Col>
            </Row>

             {result === "" ? null :
            (<Row>
              <Col className="result-container">
                <h3 id="result">{result}</h3>
              </Col>
            </Row>)

          }

          </Form>

        </div>
      </Container>
    );
  }
}

export default App;