import axios from 'axios'
const apiClient = axios.create({
    baseURL: 'http://localhost:8000'
});
const basicHeaders = {
  headers: {"Content-Type": "application/json"}
}
async function loginUser(userName, password) {
  const response = await apiClient.post(`/login`, {
      username: userName,
      password: password
    },
    basicHeaders
  );
  return response.data;
}
async function checkIfUserAuthenticated(token) {
  localStorage.getItem(token)
  const response = await apiClient.post(`/authenticate`, {}, {
      headers: {
          Authorization: `Bearer ${token}`
      }
  });
  return response.data;
}
async function createUser(userName, password) {
  const response = await apiClient.post(`/create-user`, {
      username: userName,
      password: password
  });
  return response.data;
}
async function updateUserPassword(username, oldPassword, newPassword) {
  const response = await apiClient.put(
    {
      username:username,
      password:oldPassword,
      newPassword:newPassword,
      newUsername:null
  });
  return response.data;
}
async function updateUserName(username, newUsername, password) {
  const response = await apiClient.put(
    {
      username:username,
      password:password,
      newPassword:null,
      newUsername:newUsername
  });
  return response.data;
}
async function deleteUser(username, password) {
  const response = await apiClient.put(
    {
      username:username,
      password:password
  });
  return response.data;
}
export default {
    loginUser,
    checkIfUserAuthenticated,
    createUser,
    updateUserName,
    updateUserPassword,
    deleteUser
}
