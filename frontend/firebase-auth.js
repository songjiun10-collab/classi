/*
 * Classi Firebase Auth
 * Firebase Console에서 프로젝트 만든 후 config 채우고,
 * classi_index.html의 <script> 앞에 아래를 추가:
 *   <script type="module" src="firebase-auth.js"></script>
 *
 * 그리고 doLogin/socialLogin/doLogout 함수를 Firebase 버전으로 교체.
 * 설정 방법: https://console.firebase.google.com → 프로젝트 설정 → 일반 → 내 앱
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/11.7.3/firebase-app.js";
import { getAuth, signInWithPopup, signInWithEmailAndPassword, createUserWithEmailAndPassword,
         GoogleAuthProvider, OAuthProvider, onAuthStateChanged, signOut }
  from "https://www.gstatic.com/firebasejs/11.7.3/firebase-auth.js";

const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.firebasestorage.app",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};

const fbApp = initializeApp(firebaseConfig);
const auth = getAuth(fbApp);
auth.languageCode = 'ko';

const googleProvider = new GoogleAuthProvider();
const appleProvider = new OAuthProvider('apple.com');
appleProvider.addScope('email');
appleProvider.addScope('name');
appleProvider.setCustomParameters({ locale: 'ko' });

window._fbAuth = auth;
window._googleProvider = googleProvider;
window._appleProvider = appleProvider;
window._signInWithPopup = signInWithPopup;
window._signInWithEmailAndPassword = signInWithEmailAndPassword;
window._createUserWithEmailAndPassword = createUserWithEmailAndPassword;
window._signOut = signOut;
window._firebaseReady = true;

onAuthStateChanged(auth, (user) => {
  if (user) {
    window.S.user = {
      email: user.email || '',
      name: user.displayName || user.email?.split('@')[0] || '사용자',
      role: 'user',
      provider: user.providerData[0]?.providerId || 'email',
      uid: user.uid,
      photoURL: user.photoURL
    };
    localStorage.setItem('classi-user', JSON.stringify(window.S.user));
    if (typeof showApp === 'function') showApp();
  }
});
