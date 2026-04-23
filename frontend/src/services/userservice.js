import axios from 'axios'
const apiClient = axios.create({
  baseURL: '/api'
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
    'update-password',
    {
      username:username,
      password:oldPassword,
      newPassword:newPassword,
      newUsername:""
  });
  return response.data;
}
async function updateUserName(username, newUsername, password) {
  const response = await apiClient.put(
    'update-username',
    {
      username:username,
      password:password,
      newPassword:"",
      newUsername:newUsername
  });
  return response.data;
}
async function deleteUser(username, password) {
  const response = await apiClient.post(
    'delete-user',
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
