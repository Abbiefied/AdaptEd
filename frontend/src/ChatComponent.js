import React, { useState } from 'react';
import { Button, Modal, Form, InputGroup } from 'react-bootstrap';
import axios from 'axios';

const ChatComponent = () => {
  const [show, setShow] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleShow = () => setShow(true);
  const handleClose = () => setShow(false);

  const handleSend = async () => {
    if (input.trim()) {
      const userMessage = { role: 'user', content: input };
      setMessages([...messages, userMessage]);

      try {
        const response = await axios.post(`/chat`, { message: input });
        const assistantMessage = { role: 'assistant', content: response.data.response };
        setMessages([...messages, userMessage, assistantMessage]);
      } catch (error) {
        console.error('Error sending message:', error);
      }

      setInput('');
    }
  };

  return (
    <>
      <Button onClick={handleShow} className="chat-icon">
        <img src="chat-icon.png" alt="Chat" width="50" />
      </Button>

      <Modal show={show} onHide={handleClose}>
        <Modal.Header closeButton>
          <Modal.Title>Virtual Assistant</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
          </div>
          <InputGroup className="mb-3">
            <Form.Control
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <Button variant="primary" onClick={handleSend}>
              Send
            </Button>
          </InputGroup>
        </Modal.Body>
      </Modal>
    </>
  );
};

export default ChatComponent;