"""
규칙 기반 시간표 추천기
"""
import logging
from typing import List, Dict, Optional, Set, Tuple
from ..base import Course, UserPreferences
from .base_recommender import BaseRecommender, State, Action, Reward

class RuleBasedRecommender(BaseRecommender):
    """규칙 기반 시간표 추천기"""
    
    def __init__(self, courses: List[Course], preferences: Optional[UserPreferences] = None):
        """
        Args:
            courses (List[Course]): 추천할 수 있는 과목 목록
            preferences (Optional[UserPreferences]): 사용자 선호도 설정
        """
        super().__init__(courses, preferences)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 규칙 기반 정책 설정
        self.policy = {
            'must_take_priority': 5.0,  # 필수 과목 우선순위
            'preferred_time_bonus': 2.0,  # 선호 시간대 보너스
            'rating_threshold': 3.0,  # 최소 평점 기준
            'max_courses': 8,  # 최대 과목 수
            'min_credits': 15,  # 최소 학점
            'max_credits': 21,  # 최대 학점
        }

    def _evaluate_action(self, state: State, action: Action) -> float:
        """
        행동의 가치를 평가합니다.
        
        Args:
            state (State): 현재 상태
            action (Action): 평가할 행동
            
        Returns:
            float: 행동의 가치
        """
        try:
            value = 0.0
            
            # 1. 필수 과목 우선순위
            if action.course.code in self.preferences.must_take_courses:
                value += self.policy['must_take_priority']
            
            # 2. 시간대 선호도
            if action.time_slot:
                day, start, end = action.time_slot
                if day in self.preferences.preferred_days:
                    value += self.policy['preferred_time_bonus']
            
            # 3. 평점 기준
            if action.course.rating >= self.policy['rating_threshold']:
                value += action.course.rating
            
            # 4. 학점 제한
            new_total_credits = state.total_credits + action.course.credits
            if new_total_credits > self.policy['max_credits']:
                value -= 10.0
            elif new_total_credits < self.policy['min_credits']:
                value -= 5.0
            
            # 5. 과목 수 제한
            if len(state.selected_courses) >= self.policy['max_courses']:
                value -= 15.0
            
            return value
            
        except Exception as e:
            self.logger.error(f"행동 평가 중 오류 발생: {str(e)}")
            return 0.0

    def _select_best_action(self, state: State, available_actions: List[Action]) -> Optional[Action]:
        """
        현재 상태에서 최선의 행동을 선택합니다.
        
        Args:
            state (State): 현재 상태
            available_actions (List[Action]): 가능한 행동 목록
            
        Returns:
            Optional[Action]: 선택된 행동
        """
        try:
            if not available_actions:
                return None
                
            # 각 행동의 가치 평가
            action_values = [
                (action, self._evaluate_action(state, action))
                for action in available_actions
            ]
            
            # 최고 가치의 행동 선택
            best_action, _ = max(action_values, key=lambda x: x[1])
            return best_action
            
        except Exception as e:
            self.logger.error(f"최선의 행동 선택 중 오류 발생: {str(e)}")
            return None

    def recommend_schedule(self) -> List[Course]:
        """
        규칙 기반으로 시간표를 추천합니다.
        
        Returns:
            List[Course]: 추천된 시간표
        """
        try:
            schedule: List[Course] = []
            current_state = self._get_state(schedule)
            
            while True:
                # 가능한 행동들 가져오기
                available_actions = self._get_available_actions(current_state)
                
                # 최선의 행동 선택
                best_action = self._select_best_action(current_state, available_actions)
                
                if best_action is None:
                    break
                    
                # 행동 실행
                schedule.append(best_action.course)
                
                # 상태 업데이트
                current_state = self._get_state(schedule)
                
                # 종료 조건 체크
                if (len(schedule) >= self.policy['max_courses'] or
                    current_state.total_credits >= self.policy['max_credits']):
                    break
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"시간표 추천 중 오류 발생: {str(e)}")
            return []

    def save_model(self) -> None:
        """
        규칙 기반 모델은 저장할 상태가 없습니다.
        """
        pass

    def load_model(self) -> bool:
        """
        규칙 기반 모델은 로드할 상태가 없습니다.
        
        Returns:
            bool: 항상 True 반환
        """
        return True 