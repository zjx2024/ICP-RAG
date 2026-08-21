import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import AppTheme from './theme';
import App from './App';
import './styles/glass.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppTheme>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AppTheme>
  </React.StrictMode>,
);
