"""
시간표 추천 시스템의 기본 클래스
"""
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from ..base import Course, UserPreferences

@dataclass
class State:
    """시간표 상태를 나타내는 데이터 클래스"""
    selected_courses: List[Course]  # 현재 선택된 과목들
    total_credits: int  # 현재 총 학점
    time_slots: Dict[str, Set[Tuple[int, int]]]  # 각 요일별 사용 중인 시간대
    course_types: Dict[str, int]  # 과목 유형별 개수
    years: Dict[int, int]  # 학년별 과목 개수

@dataclass
class Action:
    """행동을 나타내는 데이터 클래스"""
    course: Course  # 선택할 과목
    time_slot: Optional[Tuple[str, int, int]]  # 선택할 시간대 (요일, 시작시간, 종료시간)

@dataclass
class Reward:
    """보상을 나타내는 데이터 클래스"""
    time_conflict: float  # 시간 충돌에 대한 보상
    credit_limit: float  # 학점 제한에 대한 보상
    preference_satisfaction: float  # 선호도 만족에 대한 보상
    must_take_inclusion: float  # 필수 과목 포함에 대한 보상
    total: float  # 총 보상

@dataclass
class CourseScore:
    """과목 점수 계산을 위한 설정"""
    must_take_bonus: float = 2.0
    rating_multiplier: float = 0.5
    min_rating: float = 3.0

class BaseRecommender:
    """시간표 추천 시스템의 기본 클래스"""
    
    # 클래스 레벨 상수 정의
    COURSE_TYPE_WEIGHTS = {
        "전공필수": 1.5,
        "전공선택": 1.2,
    }
    
    YEAR_WEIGHTS = {
        1: 2.0,
        2: 2.0,
        3: 2.0,
        4: 2.0
    }
    
    def __init__(self, courses: List[Course], preferences: Optional[UserPreferences] = None):
        """
        Args:
            courses (List[Course]): 추천할 수 있는 과목 목록
            preferences (Optional[UserPreferences]): 사용자 선호도 설정
        """
        if not courses:
            raise ValueError("과목 목록이 비어있습니다.")
            
        self.courses = courses
        self.preferences = preferences or UserPreferences()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._score_config = CourseScore()

    def _calculate_course_score(self, course: Course) -> float:
        """
        과목의 점수를 계산합니다.
        
        Args:
            course (Course): 점수를 계산할 과목
            
        Returns:
            float: 계산된 과목 점수
        """
        try:
            # 제외 과목 체크
            if course.code in self.preferences.excluded_courses:
                return 0.0
                
            # 기본 점수 계산
            score = (
                self.COURSE_TYPE_WEIGHTS.get(course.course_type, 1.0) +
                self.YEAR_WEIGHTS.get(course.year, 1.0)
            )
            
            # 필수 과목 보너스
            if course.code in self.preferences.must_take_courses:
                score += self._score_config.must_take_bonus
                
            # 평점 선호도 반영
            if course.rating >= self.preferences.min_rating:
                score += (course.rating - self.preferences.min_rating) * self._score_config.rating_multiplier
                
            return score
            
        except Exception as e:
            self.logger.error(f"과목 점수 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _check_time_slot_conflict(self, slot1: Tuple[str, int, int], slot2: Tuple[str, int, int]) -> bool:
        """
        두 시간 슬롯 간의 충돌을 확인합니다.
        
        Args:
            slot1 (Tuple[str, int, int]): (요일, 시작시간, 종료시간)
            slot2 (Tuple[str, int, int]): (요일, 시작시간, 종료시간)
            
        Returns:
            bool: 시간 충돌 여부
        """
        day1, start1, end1 = slot1
        day2, start2, end2 = slot2
        
        return (day1 == day2 and 
                start1 <= end2 and 
                start2 <= end1)

    def check_time_conflict(self, course1: Course, course2: Course) -> bool:
        """
        두 과목 간의 시간 충돌을 확인합니다.
        
        Args:
            course1 (Course): 첫 번째 과목
            course2 (Course): 두 번째 과목
            
        Returns:
            bool: 시간 충돌 여부
        """
        try:
            for slot1 in course1.time_slots:
                for slot2 in course2.time_slots:
                    if self._check_time_slot_conflict(
                        (slot1.day, slot1.start_time, slot1.end_time),
                        (slot2.day, slot2.start_time, slot2.end_time)
                    ):
                        return True
            return False
            
        except Exception as e:
            self.logger.error(f"시간 충돌 확인 중 오류 발생: {str(e)}")
            return True  # 오류 발생 시 안전하게 충돌로 처리

    def calculate_total_credits(self, schedule: List[Course]) -> int:
        """
        시간표의 총 학점을 계산합니다.
        
        Args:
            schedule (List[Course]): 학점을 계산할 시간표
            
        Returns:
            int: 총 학점
        """
        try:
            return sum(course.credits for course in schedule) if schedule else 0
            
        except Exception as e:
            self.logger.error(f"총 학점 계산 중 오류 발생: {str(e)}")
            return 0

    def recommend_schedule(self) -> List[Course]:
        """
        시간표를 추천합니다.
        
        Returns:
            List[Course]: 추천된 시간표
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def save_model(self) -> None:
        """
        모델을 저장합니다.
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def load_model(self) -> bool:
        """
        모델을 로드합니다.
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")